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

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
use super::ErasedTensorInner;
use super::{DimDyn, Dimension, Tensor, TensorInner, TensorOp};
use crate::ast::DType;
use crate::backend::Buffer;
use crate::backend::global::{DeviceKind, get_default_device_kind};
use crate::tensor::shape::{Expr, View};
use ndarray::{Array, ArrayD, Dimension as NdDimension, IxDyn};
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

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
        }
    }
}

impl std::error::Error for ForwardError {}

impl<D: Dimension> Tensor<f32, D> {
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
        // If already executed, return self
        if self.is_executed() {
            return Ok(self);
        }

        let device_kind = get_default_device_kind();

        match device_kind {
            DeviceKind::None => Err(ForwardError::NoDefaultDevice),

            #[cfg(all(feature = "metal", target_os = "macos"))]
            DeviceKind::Metal => self.realize_metal(),

            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            DeviceKind::Metal => Err(ForwardError::DeviceUnavailable(
                "Metal backend is not available. Enable 'metal' feature on macOS.".to_string(),
            )),

            #[cfg(feature = "opencl")]
            DeviceKind::OpenCL => self.realize_opencl(),

            #[cfg(not(feature = "opencl"))]
            DeviceKind::OpenCL => Err(ForwardError::DeviceUnavailable(
                "OpenCL backend is not available. Enable 'opencl' feature.".to_string(),
            )),
        }
    }

    /// Collect input tensor data from the computation graph
    ///
    /// Returns input data in the order they appear as non-const inputs in the Compute ops.
    /// This matches the order used by the lowerer when generating `input0`, `input1`, etc.
    #[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
    fn collect_input_data(&self) -> Vec<(Vec<f32>, Vec<usize>)> {
        use std::collections::HashSet;

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
            inner: &dyn ErasedTensorInner,
            visited: &mut HashSet<usize>,
            inputs: &mut Vec<(Vec<f32>, Vec<usize>)>,
        ) {
            let ptr = inner as *const dyn ErasedTensorInner as *const () as usize;
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            match inner.op() {
                // Leaf nodes with data
                TensorOp::Executed => {
                    // Get data from buffer (now Buffer)
                    if let Ok(guard) = inner.buffer().read()
                        && let Some(buf) = guard.as_ref()
                        && let Ok(bytes) = buf.read_to_host()
                    {
                        let data = bytes_to_f32(&bytes);
                        inputs.push((data, inner.shape().to_vec()));
                    }
                }
                TensorOp::Buffer { .. } => {
                    // Named buffer - also get data if available
                    if let Ok(guard) = inner.buffer().read()
                        && let Some(buf) = guard.as_ref()
                        && let Ok(bytes) = buf.read_to_host()
                    {
                        let data = bytes_to_f32(&bytes);
                        inputs.push((data, inner.shape().to_vec()));
                    }
                }
                // Compute operations - recurse into inputs
                TensorOp::Compute { .. } => {
                    for input in inner.op().inputs() {
                        // Skip const inputs as the lowerer embeds them directly
                        if !matches!(input.op(), TensorOp::Const(_)) {
                            collect_recursive(input.as_ref(), visited, inputs);
                        }
                    }
                }
                // Other operations - recurse into inputs
                _ => {
                    for input in inner.op().inputs() {
                        collect_recursive(input.as_ref(), visited, inputs);
                    }
                }
            }
        }

        let mut visited = HashSet::new();
        let mut inputs = Vec::new();
        collect_recursive(self.inner.as_ref(), &mut visited, &mut inputs);
        inputs
    }

    /// Internal: Execute realize on Metal device
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn realize_metal(&self) -> Result<&Self, ForwardError> {
        use crate::backend::cache::{
            KernelCacheKey, MetalCacheEntry, get_metal_kernel, insert_metal_kernel,
        };
        use crate::backend::global::get_default_device;
        use crate::backend::metal::{MetalBuffer, MetalCompiler, MetalDevice};
        use crate::backend::traits::TypedBuffer;
        use crate::backend::{BufferSignature, CompiledKernel, KernelSignature, Pipeline};
        use crate::renderer::MetalRenderer;
        use crate::tensor::lowerer::TensorLowerer;
        use crate::tensor::stringify::stringify_graph;
        use std::time::Instant;

        // Get the Metal device
        let device: std::sync::Arc<MetalDevice> = get_default_device::<MetalDevice>()
            .ok_or_else(|| ForwardError::DeviceUnavailable("Metal device not found".to_string()))?;

        // Generate cache key from graph structure (include device identity)
        let graph_repr = stringify_graph(self);
        let device_id = Arc::as_ptr(&device) as usize;
        let cache_key = KernelCacheKey::new(graph_repr, DeviceKind::Metal, device_id);

        // Check cache for compiled kernel
        let compiled: CompiledKernel<_, MetalBuffer> =
            if let Some(cached) = get_metal_kernel(&cache_key) {
                log::debug!("Kernel cache hit: {}", cache_key.graph_repr());
                CompiledKernel::new(cached.kernel, cached.signature, cached.dispatch_config)
            } else {
                log::debug!("Kernel cache miss: {}", cache_key.graph_repr());

                // Collect input tensor data
                let input_data = self.collect_input_data();

                // Lower Tensor to AST directly
                let mut lowerer = TensorLowerer::new();
                let ast = lowerer.lower(&self.clone().into_dyn());

                // Create input signatures
                let input_signatures: Vec<BufferSignature> = input_data
                    .iter()
                    .enumerate()
                    .map(|(i, (_, shape))| {
                        let shape_expr: Vec<Expr> =
                            shape.iter().map(|&s| Expr::from(s as i64)).collect();
                        BufferSignature::new(format!("input{}", i), shape_expr)
                    })
                    .collect();

                // Create signature from tensor metadata
                let output_shape_expr: Vec<Expr> =
                    self.shape().iter().map(|&s| Expr::from(s as i64)).collect();
                let signature = KernelSignature::new(
                    input_signatures,
                    vec![BufferSignature::new(
                        "output".to_string(),
                        output_shape_expr,
                    )],
                );

                // Create pipeline and compile from AST
                use crate::backend::traits::Compiler;
                let renderer = MetalRenderer::default();
                let compiler = MetalCompiler::new();
                let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
                    Pipeline::new(renderer, compiler, device.as_ref().clone());

                let compiled = pipeline.compile_ast(ast, signature).map_err(|e| {
                    ForwardError::CompilationError(format!("Failed to compile: {:?}", e))
                })?;

                // Insert into cache
                insert_metal_kernel(
                    cache_key,
                    MetalCacheEntry {
                        kernel: compiled.kernel.clone(),
                        signature: compiled.signature.clone(),
                        dispatch_config: compiled.dispatch_config.clone(),
                        last_accessed: Instant::now(),
                    },
                );

                compiled
            };

        // Collect input tensor data (needed for buffer creation)
        let input_data = self.collect_input_data();

        // Create input buffers from collected data
        use crate::backend::traits::Buffer;
        let mut input_buffers: Vec<MetalBuffer> = Vec::new();
        for (data, shape) in &input_data {
            let buffer = MetalBuffer::from_vec(device.as_ref(), shape.clone(), DType::F32, data)
                .map_err(|e| {
                    ForwardError::ExecutionError(format!("Failed to create input buffer: {}", e))
                })?;
            input_buffers.push(buffer);
        }

        // Allocate output buffer
        let output_shape = self.shape().to_vec();
        let mut output_buffer = MetalBuffer::allocate(device.as_ref(), output_shape, DType::F32)
            .map_err(|e| {
                ForwardError::ExecutionError(format!("Failed to create output buffer: {}", e))
            })?;

        // Execute kernel with input and output buffers
        let input_refs: Vec<&MetalBuffer> = input_buffers.iter().collect();
        let mut output_refs: Vec<&mut MetalBuffer> = vec![&mut output_buffer];

        compiled
            .execute(&input_refs, &mut output_refs)
            .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

        // Store GPU buffer directly in self
        if let Ok(mut guard) = self.inner.buffer.write() {
            *guard = Some(Box::new(output_buffer) as Box<dyn Buffer>);
        }

        Ok(self)
    }

    /// Internal: Execute realize on OpenCL device
    #[cfg(feature = "opencl")]
    fn realize_opencl(&self) -> Result<&Self, ForwardError> {
        use crate::backend::cache::{
            KernelCacheKey, OpenCLCacheEntry, get_opencl_kernel, insert_opencl_kernel,
        };
        use crate::backend::global::get_default_device;
        use crate::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice};
        use crate::backend::traits::TypedBuffer;
        use crate::backend::{BufferSignature, CompiledKernel, KernelSignature, Pipeline};
        use crate::renderer::OpenCLRenderer;
        use crate::tensor::lowerer::TensorLowerer;
        use crate::tensor::stringify::stringify_graph;
        use std::time::Instant;

        // Get the OpenCL device
        let device: std::sync::Arc<OpenCLDevice> = get_default_device::<OpenCLDevice>()
            .ok_or_else(|| {
                ForwardError::DeviceUnavailable("OpenCL device not found".to_string())
            })?;

        // Generate cache key from graph structure (include device identity)
        let graph_repr = stringify_graph(self);
        let device_id = Arc::as_ptr(&device) as usize;
        let cache_key = KernelCacheKey::new(graph_repr, DeviceKind::OpenCL, device_id);

        // Check cache for compiled kernel
        let compiled: CompiledKernel<_, OpenCLBuffer> =
            if let Some(cached) = get_opencl_kernel(&cache_key) {
                log::debug!("Kernel cache hit: {}", cache_key.graph_repr());
                CompiledKernel::new(cached.kernel, cached.signature, cached.dispatch_config)
            } else {
                log::debug!("Kernel cache miss: {}", cache_key.graph_repr());

                // Collect input tensor data
                let input_data = self.collect_input_data();

                // Lower Tensor to AST directly
                let mut lowerer = TensorLowerer::new();
                let ast = lowerer.lower(&self.clone().into_dyn());

                // Create input signatures
                let input_signatures: Vec<BufferSignature> = input_data
                    .iter()
                    .enumerate()
                    .map(|(i, (_, shape))| {
                        let shape_expr: Vec<Expr> =
                            shape.iter().map(|&s| Expr::from(s as i64)).collect();
                        BufferSignature::new(format!("input{}", i), shape_expr)
                    })
                    .collect();

                // Create signature from tensor metadata
                let output_shape_expr: Vec<Expr> =
                    self.shape().iter().map(|&s| Expr::from(s as i64)).collect();
                let signature = KernelSignature::new(
                    input_signatures,
                    vec![BufferSignature::new(
                        "output".to_string(),
                        output_shape_expr,
                    )],
                );

                // Create pipeline and compile from AST
                use crate::backend::traits::Compiler;
                let renderer = OpenCLRenderer::default();
                let compiler = OpenCLCompiler::new();
                let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
                    Pipeline::new(renderer, compiler, device.as_ref().clone());

                let compiled = pipeline.compile_ast(ast, signature).map_err(|e| {
                    ForwardError::CompilationError(format!("Failed to compile: {:?}", e))
                })?;

                // Insert into cache
                insert_opencl_kernel(
                    cache_key,
                    OpenCLCacheEntry {
                        kernel: compiled.kernel.clone(),
                        signature: compiled.signature.clone(),
                        dispatch_config: compiled.dispatch_config.clone(),
                        last_accessed: Instant::now(),
                    },
                );

                compiled
            };

        // Collect input tensor data (needed for buffer creation)
        let input_data = self.collect_input_data();

        // Create input buffers from collected data
        use crate::backend::traits::Buffer;
        let mut input_buffers: Vec<OpenCLBuffer> = Vec::new();
        for (data, shape) in &input_data {
            let buffer = OpenCLBuffer::from_vec(device.as_ref(), shape.clone(), DType::F32, data)
                .map_err(|e| {
                ForwardError::ExecutionError(format!("Failed to create input buffer: {}", e))
            })?;
            input_buffers.push(buffer);
        }

        // Allocate output buffer
        let output_shape = self.shape().to_vec();
        let mut output_buffer = OpenCLBuffer::allocate(device.as_ref(), output_shape, DType::F32)
            .map_err(|e| {
            ForwardError::ExecutionError(format!("Failed to create output buffer: {}", e))
        })?;

        // Execute kernel with input and output buffers
        let input_refs: Vec<&OpenCLBuffer> = input_buffers.iter().collect();
        let mut output_refs: Vec<&mut OpenCLBuffer> = vec![&mut output_buffer];

        compiled
            .execute(&input_refs, &mut output_refs)
            .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

        // Store GPU buffer directly in self
        if let Ok(mut guard) = self.inner.buffer.write() {
            *guard = Some(Box::new(output_buffer) as Box<dyn Buffer>);
        }

        Ok(self)
    }

    /// Create an executed tensor from raw data
    ///
    /// This creates a tensor with the buffer already populated,
    /// bypassing the computation graph.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32, DimDyn> {
        let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::contiguous(shape_exprs);
        let vec_buffer = VecBuffer::from_vec(&data, shape.clone(), DType::F32);
        let inner = TensorInner {
            op: TensorOp::Executed,
            view,
            shape,
            dtype: DType::F32,
            name: None,
            autograd: None,
            buffer: RwLock::new(Some(Box::new(vec_buffer) as Box<dyn Buffer>)),
        };

        Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Get the computed data from the tensor
    ///
    /// Returns the data if the tensor has been executed (via contiguous() or forward()).
    ///
    /// Returns None if the tensor has not been executed yet.
    pub fn data(&self) -> Option<Vec<f32>> {
        // Check the buffer field in TensorInner
        if let Ok(guard) = self.inner.buffer.read()
            && let Some(buf) = guard.as_ref()
            && let Ok(bytes) = buf.read_to_host()
        {
            // Convert bytes to Vec<f32>
            let len = bytes.len() / std::mem::size_of::<f32>();
            let mut result = Vec::with_capacity(len);
            unsafe {
                let ptr = bytes.as_ptr() as *const f32;
                result.extend_from_slice(std::slice::from_raw_parts(ptr, len));
            }
            return Some(result);
        }
        None
    }

    // ========================================================================
    // ndarray conversion methods
    // ========================================================================

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

impl Tensor<f32, DimDyn> {
    /// Create a tensor from an ndarray with any dimensions
    ///
    /// This creates a DimDyn tensor with the buffer already populated,
    /// bypassing the computation graph.
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::array;
    /// let arr = array![[1.0, 2.0], [3.0, 4.0]];
    /// let t = Tensor::<f32, DimDyn>::from_ndarray(&arr);
    /// assert_eq!(t.shape(), &[2, 2]);
    /// ```
    pub fn from_ndarray<ND: NdDimension>(array: &Array<f32, ND>) -> Tensor<f32, DimDyn> {
        let shape: Vec<usize> = array.shape().to_vec();
        let data: Vec<f32> = array
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| array.iter().cloned().collect());
        Self::from_data(data, shape)
    }
}

// Type-safe from_ndarray implementations using macro
macro_rules! impl_from_ndarray {
    ($dim:ty, $ix:ty, $n:expr) => {
        impl Tensor<f32, $dim> {
            /// Create a tensor from an ndarray with type-safe dimensions
            ///
            /// This creates a tensor with the buffer already populated,
            /// bypassing the computation graph. The ndarray dimension type
            /// must match the tensor dimension type.
            pub fn from_ndarray(array: &Array<f32, $ix>) -> Tensor<f32, $dim> {
                use super::{TensorInner, TensorOp};
                use crate::tensor::shape::{Expr, View};
                use std::sync::RwLock;

                let shape: Vec<usize> = array.shape().to_vec();
                assert_eq!(
                    shape.len(),
                    $n,
                    "Array dimension mismatch: expected {}, got {}",
                    $n,
                    shape.len()
                );

                let data: Vec<f32> = array
                    .as_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| array.iter().cloned().collect());

                let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
                let view = View::contiguous(shape_exprs);
                let vec_buffer = VecBuffer::from_vec(&data, shape.clone(), DType::F32);
                let inner = TensorInner {
                    op: TensorOp::Executed,
                    view,
                    shape,
                    dtype: DType::F32,
                    name: None,
                    autograd: None,
                    buffer: RwLock::new(Some(Box::new(vec_buffer) as Box<dyn Buffer>)),
                };

                Tensor {
                    inner: Arc::new(inner),
                    _dtype: PhantomData,
                    _dim: PhantomData,
                }
            }
        }
    };
}

use super::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6};
use ndarray::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};

impl_from_ndarray!(Dim0, Ix0, 0);
impl_from_ndarray!(Dim1, Ix1, 1);
impl_from_ndarray!(Dim2, Ix2, 2);
impl_from_ndarray!(Dim3, Ix3, 3);
impl_from_ndarray!(Dim4, Ix4, 4);
impl_from_ndarray!(Dim5, Ix5, 5);
impl_from_ndarray!(Dim6, Ix6, 6);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim1, Dim2, Dim3};
    use ndarray::{Array1, Array2, Array3, array};

    // ========================================================================
    // Type-safe conversion tests
    // ========================================================================

    #[test]
    fn test_from_ndarray_dim1_type_safe() {
        let arr: Array1<f32> = array![1.0, 2.0, 3.0];
        let tensor = Tensor::<f32, Dim1>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.data(), Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_ndarray_dim2_type_safe() {
        let arr: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let tensor = Tensor::<f32, Dim2>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_from_ndarray_dim3_type_safe() {
        let arr: Array3<f32> = Array3::zeros((2, 3, 4));
        let tensor = Tensor::<f32, Dim3>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_to_ndarray_dim2_type_safe() {
        let arr: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let tensor = Tensor::<f32, Dim2>::from_ndarray(&arr);

        // Type-safe: returns Array2<f32> automatically
        let recovered: Array2<f32> = tensor.to_ndarray().unwrap();
        assert_eq!(recovered.shape(), &[2, 3]);
        assert_eq!(recovered[[0, 0]], 1.0);
        assert_eq!(recovered[[1, 2]], 6.0);
    }

    #[test]
    fn test_roundtrip_dim2_type_safe() {
        let original: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Convert to Tensor<f32, Dim2>
        let tensor = Tensor::<f32, Dim2>::from_ndarray(&original);

        // Convert back - type-safe!
        let recovered: Array2<f32> = tensor.to_ndarray().unwrap();

        // Verify they match
        assert_eq!(original, recovered);
    }

    // ========================================================================
    // DimDyn (dynamic dimension) tests
    // ========================================================================

    #[test]
    fn test_from_ndarray_dimdyn() {
        let arr = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::<f32, DimDyn>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_to_ndarray_dimdyn() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<f32, DimDyn>::from_data(data.clone(), vec![2, 3]);

        // DimDyn returns ArrayD (IxDyn)
        let arr = tensor.to_ndarray().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
    }

    #[test]
    fn test_to_ndarray_dyn_explicit() {
        let arr: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let tensor = Tensor::<f32, Dim2>::from_ndarray(&arr);

        // Can also get dynamic array from static tensor
        let dyn_arr = tensor.to_ndarray_dyn().unwrap();
        assert_eq!(dyn_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_to_ndarray_not_executed() {
        // Create a tensor that hasn't been executed
        let tensor = Tensor::<f32, Dim2>::input("x", [2, 3]);
        // Should return None since it's not executed
        assert!(tensor.to_ndarray().is_none());
    }
}
