//! Forward execution for Tensor
//!
//! This module provides the realize() implementation which compiles and
//! executes the lazy computation graph on the default device.
//!
//! ## Key Methods
//!
//! - `realize()`: The primary execution trigger (like tinygrad). Compiles the
//!   computation graph, executes it on the device, and stores the result in self.
//! - `data()`: Returns the computed data if available.
//! - `from_data()`: Creates a tensor with pre-populated buffer data.
//!
//! ## Backend Integration
//!
//! This module uses a callback-based architecture for backend integration.
//! Backend crates (e.g., harp-backend-opencl) register their realize implementations
//! via `register_realizer()`.

use super::{DimDyn, Dimension, FloatDType, Tensor, TensorDType, TensorInner, TensorOp};
use crate::ast::DType;
use crate::backend::Buffer;
use crate::backend::global::{DeviceKind, get_default_device_kind, has_default_device};
use crate::tensor::shape::{Expr, View};
use ndarray::{Array, ArrayD, Dimension as NdDimension, IxDyn};
use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock, RwLock};

// ============================================================================
// VecBuffer - Simple wrapper for host data to implement Buffer
// ============================================================================

/// Simple wrapper for Vec<u8> that implements Buffer for host data storage
///
/// This is used by `from_data()` and for input tensors that haven't been
/// transferred to the GPU yet.
#[derive(Clone)]
pub(crate) struct VecBuffer {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
}

impl VecBuffer {
    /// Create a new VecBuffer from typed data
    pub fn from_vec<T>(data: &[T], shape: Vec<usize>, dtype: DType) -> Self {
        let byte_len = std::mem::size_of_val(data);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        Self {
            data: bytes.to_vec(),
            shape,
            dtype,
        }
    }
}

impl Buffer for VecBuffer {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn byte_len(&self) -> usize {
        self.data.len()
    }

    fn read_to_host(&self) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.data.clone())
    }

    fn write_from_host(
        &mut self,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.data = data.to_vec();
        Ok(())
    }

    fn clone_buffer(&self) -> Box<dyn Buffer> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Error type for forward execution
#[derive(Debug, Clone)]
pub enum ForwardError {
    /// No default device is set
    NoDefaultDevice,
    /// The requested device is not available (feature not enabled or device not found)
    DeviceUnavailable(String),
    /// Failed to compile the computation graph
    CompilationError(String),
    /// Failed to execute the kernel
    ExecutionError(String),
    /// Missing input data for a tensor that requires it
    MissingInputData(String),
    /// The tensor has no computation to execute (e.g., just a buffer)
    NoComputation,
    /// The tensor is already executed (buffer is populated)
    AlreadyExecuted,
    /// No realizer registered for the device kind
    NoRealizer(DeviceKind),
}

impl fmt::Display for ForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForwardError::NoDefaultDevice => {
                write!(f, "No default device set. Use set_default_device() first.")
            }
            ForwardError::DeviceUnavailable(msg) => write!(f, "Device unavailable: {}", msg),
            ForwardError::CompilationError(msg) => write!(f, "Compilation error: {}", msg),
            ForwardError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            ForwardError::MissingInputData(msg) => write!(f, "Missing input data: {}", msg),
            ForwardError::NoComputation => {
                write!(f, "Tensor has no computation graph to execute")
            }
            ForwardError::AlreadyExecuted => {
                write!(f, "Tensor is already executed (buffer is populated)")
            }
            ForwardError::NoRealizer(kind) => {
                write!(
                    f,
                    "No realizer registered for device kind {:?}. Make sure the backend crate is included.",
                    kind
                )
            }
        }
    }
}

impl std::error::Error for ForwardError {}

// ============================================================================
// Realizer callback registry
// ============================================================================

/// Type alias for realizer callback function
///
/// This function is called by `realize_core` to execute the computation graph
/// on a specific device. Backend crates register their implementations via
/// `register_realizer()`.
pub type RealizerCallback = fn(&TensorInner) -> Result<(), ForwardError>;

/// Entry in the realizer registry
struct RealizerEntry {
    device_kind: DeviceKind,
    callback: RealizerCallback,
}

/// Global registry for realizer callbacks
static REALIZERS: OnceLock<RwLock<Vec<RealizerEntry>>> = OnceLock::new();

/// Register a realizer callback for a specific device kind
///
/// Backend crates call this function during initialization to register their
/// realize implementation.
///
/// # Example
///
/// ```ignore
/// // In harp-backend-opencl/src/lib.rs
/// fn init() {
///     harp_core::tensor::forward::register_realizer(
///         DeviceKind::OpenCL,
///         realize_opencl,
///     );
/// }
/// ```
pub fn register_realizer(device_kind: DeviceKind, callback: RealizerCallback) {
    let realizers = REALIZERS.get_or_init(|| RwLock::new(Vec::new()));
    let mut realizers = realizers.write().unwrap();

    // Don't register duplicates
    if !realizers.iter().any(|r| r.device_kind == device_kind) {
        log::debug!("Registering realizer for {:?}", device_kind);
        realizers.push(RealizerEntry {
            device_kind,
            callback,
        });
    }
}

/// Get the realizer callback for a device kind
fn get_realizer(device_kind: DeviceKind) -> Option<RealizerCallback> {
    let realizers = REALIZERS.get_or_init(|| RwLock::new(Vec::new()));
    let realizers = realizers.read().unwrap();
    realizers
        .iter()
        .find(|r| r.device_kind == device_kind)
        .map(|r| r.callback)
}

// ============================================================================
// collect_input_data_inner - Standalone function for realize implementations
// ============================================================================

/// Collect input tensor data from a computation graph
///
/// Returns input data in the order they appear as non-const inputs in the Compute ops.
/// This matches the order used by the lowerer when generating `input0`, `input1`, etc.
///
/// This is a public utility function that backend crates can use in their
/// realizer implementations.
pub fn collect_input_data_inner(inner: &TensorInner) -> Vec<(Vec<f32>, Vec<usize>)> {
    /// Convert bytes to Vec<f32>
    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        let len = bytes.len() / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(len);
        unsafe {
            let ptr = bytes.as_ptr() as *const f32;
            result.extend_from_slice(std::slice::from_raw_parts(ptr, len));
        }
        result
    }

    fn collect_recursive(
        inner: &TensorInner,
        visited: &mut HashSet<usize>,
        inputs: &mut Vec<(Vec<f32>, Vec<usize>)>,
    ) {
        let ptr = inner as *const TensorInner as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // バッファを持つノードはすべて入力として扱う（ConstFill, Compute等も含む）
        // これにより collect_input_buffers (lowerer) と同じ順序で入力を収集する
        if inner.has_buffer() {
            log::debug!(
                "collect_input_data: adding buffer for {} (shape={:?})",
                inner.op.name(),
                inner.shape()
            );
            if let Some(bytes) = inner.read_buffer() {
                let data = bytes_to_f32(&bytes);
                inputs.push((data, inner.shape().to_vec()));
            }
            return;
        }

        match inner.op() {
            // Const/ConstFill are embedded directly by the lowerer, skip them
            TensorOp::Const(_) | TensorOp::ConstFill(_) => {}
            // Other operations - recurse into inputs
            _ => {
                for input in inner.op().inputs() {
                    // Skip const inputs as the lowerer embeds them directly
                    if !matches!(input.op(), TensorOp::Const(_) | TensorOp::ConstFill(_)) {
                        collect_recursive(input.as_ref(), visited, inputs);
                    }
                }
            }
        }
    }

    let mut visited = HashSet::new();
    let mut inputs = Vec::new();
    collect_recursive(inner, &mut visited, &mut inputs);
    inputs
}

// ============================================================================
// TensorInner realize_core - Core realize implementation
// ============================================================================

impl TensorInner {
    /// Core realize processing (callable from &self)
    ///
    /// This is the internal implementation of realize that dispatches to
    /// the appropriate backend's realizer callback.
    pub(crate) fn realize_core(&self) -> Result<(), ForwardError> {
        // If already executed (has buffer), skip
        if self.buffer.read().unwrap().is_some() {
            return Ok(());
        }

        // Check if a device is set
        if !has_default_device() {
            return Err(ForwardError::NoDefaultDevice);
        }

        let device_kind = get_default_device_kind();

        match device_kind {
            DeviceKind::None => Err(ForwardError::NoDefaultDevice),

            DeviceKind::C => Err(ForwardError::DeviceUnavailable(
                "C backend does not support runtime execution. Use Metal or OpenCL for GPU execution.".to_string(),
            )),

            kind => {
                // Get the registered realizer for this device kind
                let realizer = get_realizer(kind).ok_or(ForwardError::NoRealizer(kind))?;

                // Call the realizer
                realizer(self)
            }
        }
    }
}

// ============================================================================
// realize - Generic over TensorDType
// ============================================================================

impl<T: TensorDType, D: Dimension> Tensor<T, D> {
    /// Execute the computation graph and store the result in self
    ///
    /// This is the primary execution trigger for lazy evaluation (like tinygrad's `.realize()`).
    /// It:
    /// 1. Checks if already executed (returns self if so)
    /// 2. Builds a Graph from the internal computation graph
    /// 3. Compiles and optimizes the graph using Pipeline
    /// 4. Executes the kernel on the device
    /// 5. Stores the computed buffer in self
    ///
    /// Each realize() call = one kernel execution.
    ///
    /// # Returns
    ///
    /// A reference to self, allowing method chaining.
    ///
    /// # Errors
    ///
    /// Returns `ForwardError` if:
    /// - No default device is set
    /// - The device backend is not available (feature not enabled)
    /// - Compilation fails
    /// - Execution fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use harp::tensor::{Tensor, Dim2};
    /// use harp::backend::{set_default_device, DeviceKind};
    ///
    /// // Set up device
    /// let device = ...; // Get Metal or OpenCL device
    /// set_default_device(device, DeviceKind::Metal);
    ///
    /// // Create lazy computation
    /// let a = Tensor::<f32, Dim2>::full([3, 4], 1.0);
    /// let b = Tensor::<f32, Dim2>::full([3, 4], 2.0);
    /// let c = &a + &b;
    ///
    /// // Execute computation
    /// c.realize()?;
    ///
    /// // Get result data
    /// let data = c.data().unwrap();
    /// ```
    pub fn realize(&self) -> Result<&Self, ForwardError> {
        // Use realize_recursive to automatically realize all input tensors first
        // This enables "auto-realize" of intermediate results
        self.inner
            .realize_recursive()
            .map_err(ForwardError::ExecutionError)?;
        Ok(self)
    }
}

// ============================================================================
// data - Generic over FloatDType and Dimension
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Get the computed data from the tensor
    ///
    /// Returns the data if the tensor has been executed (via contiguous() or forward()).
    ///
    /// Returns None if the tensor has not been executed yet.
    pub fn data(&self) -> Option<Vec<T>> {
        Self::data_inner(&self.inner)
    }

    /// Internal helper to get data from a TensorInner, handling Views recursively
    fn data_inner(inner: &TensorInner) -> Option<Vec<T>> {
        // Check the buffer field in TensorInner
        if let Ok(guard) = inner.buffer.read()
            && let Some(buf) = guard.as_ref()
            && let Ok(bytes) = buf.read_to_host()
        {
            // Convert bytes to Vec<T>
            let len = bytes.len() / std::mem::size_of::<T>();
            let mut result = Vec::with_capacity(len);
            unsafe {
                let ptr = bytes.as_ptr() as *const T;
                result.extend_from_slice(std::slice::from_raw_parts(ptr, len));
            }
            return Some(result);
        }

        // For View operations, get data from the underlying tensor
        // Note: This returns the raw underlying data. For non-contiguous views,
        // the caller may need to reinterpret according to the view's strides.
        if let TensorOp::View { input } = inner.op() {
            return Self::data_inner(input.as_ref());
        }

        None
    }
}

// ============================================================================
// from_data - Generic over FloatDType (DimDyn only)
// ============================================================================

impl<T: FloatDType> Tensor<T, DimDyn> {
    /// Create an executed tensor from raw data
    ///
    /// This creates a tensor with the buffer already populated,
    /// bypassing the computation graph.
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Tensor<T, DimDyn> {
        let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::contiguous(shape_exprs);
        let vec_buffer = VecBuffer::from_vec(&data, shape.clone(), T::DTYPE);
        let inner = TensorInner {
            op: TensorOp::Executed,
            view,
            shape,
            dtype: T::DTYPE,
            name: None,
            buffer: RwLock::new(Some(Box::new(vec_buffer) as Box<dyn Buffer>)),
        };

        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// ndarray conversion methods - f32 only
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Convert the tensor to an ndarray with dynamic dimensions
    ///
    /// Returns the data as an ndarray if the tensor has been executed.
    /// Returns None if the tensor has not been executed yet.
    ///
    /// # Example
    /// ```ignore
    /// let t = Tensor::<f32, Dim2>::full([2, 3], 1.0);
    /// t.forward()?;
    /// let arr = t.to_ndarray_dyn().unwrap();
    /// assert_eq!(arr.shape(), &[2, 3]);
    /// ```
    pub fn to_ndarray_dyn(&self) -> Option<ArrayD<f32>> {
        let data = self.data()?;
        let shape = IxDyn(self.shape());
        Some(Array::from_shape_vec(shape, data).expect("Shape mismatch in to_ndarray_dyn"))
    }

    /// Convert the tensor to an ndarray with type-safe dimensions
    ///
    /// Returns the data as an ndarray with the dimension type matching the tensor's
    /// dimension type. This is type-safe: Tensor<f32, Dim2> returns Array<f32, Ix2>, etc.
    ///
    /// Returns None if the tensor has not been executed yet.
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::Ix2;
    ///
    /// let t = Tensor::<f32, Dim2>::full([2, 3], 1.0);
    /// t.forward()?;
    ///
    /// // Type-safe: returns Array<f32, Ix2> automatically
    /// let arr = t.to_ndarray().unwrap();
    /// assert_eq!(arr.shape(), &[2, 3]);
    /// ```
    pub fn to_ndarray(&self) -> Option<Array<f32, D::NdArrayDim>> {
        let data = self.data()?;
        let shape = D::NdArrayDim::from_dimension(&IxDyn(self.shape()))
            .expect("Dimension mismatch in to_ndarray");
        Some(Array::from_shape_vec(shape, data).expect("Shape mismatch in to_ndarray"))
    }
}
