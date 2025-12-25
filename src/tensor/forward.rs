//! Forward execution for Tensor
//!
//! This module provides the forward() and realize() implementations which
//! compile and execute the lazy computation graph on the default device.
//!
//! ## Key Methods
//!
//! - `realize()`: The primary execution trigger (like tinygrad). Compiles the
//!   computation graph and executes it on the device, returning a new tensor
//!   with the result.
//! - `forward()`: Legacy method that stores result in autograd's cached_data.
//! - `data()`: Returns the computed data if available.
//! - `from_data()`: Creates a tensor with pre-populated buffer data.

use super::{DimDyn, Dimension, Tensor, TensorInner, TensorOp};
use crate::ast::DType;
use crate::backend::global::{DeviceKind, get_default_device_kind};
use crate::tensor::shape::{Expr, View};
use ndarray::{Array, ArrayD, Dimension as NdDimension, IxDyn};
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

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

impl<D: Dimension> Tensor<D> {
    /// Execute the computation graph and return a new tensor with the result
    ///
    /// This is the primary execution trigger for lazy evaluation (like tinygrad's `.realize()`).
    /// It:
    /// 1. Checks if already executed (returns self if so)
    /// 2. Builds a Graph from the internal computation graph
    /// 3. Compiles and optimizes the graph using Pipeline
    /// 4. Executes the kernel on the device
    /// 5. Returns a new Tensor with the computed buffer
    ///
    /// Each realize() call = one kernel execution.
    ///
    /// # Returns
    ///
    /// A new Tensor with the computed result stored in its buffer.
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
    /// let a = Tensor::<Dim2>::full([3, 4], 1.0);
    /// let b = Tensor::<Dim2>::full([3, 4], 2.0);
    /// let c = &a + &b;
    ///
    /// // Execute computation
    /// let result = c.realize()?;
    ///
    /// // Get result data
    /// let data = result.data().unwrap();
    /// ```
    pub fn realize(&self) -> Result<Tensor<D>, ForwardError> {
        // If already executed, return a clone
        if self.is_executed() {
            return Ok(self.clone());
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
    fn collect_input_data(&self) -> Vec<(Vec<f32>, Vec<usize>)> {
        use std::collections::HashSet;

        fn collect_recursive(
            inner: &Arc<TensorInner>,
            visited: &mut HashSet<usize>,
            inputs: &mut Vec<(Vec<f32>, Vec<usize>)>,
        ) {
            let ptr = Arc::as_ptr(inner) as usize;
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            match &inner.op {
                // Leaf nodes with data
                TensorOp::Executed => {
                    // Get data from buffer
                    if let Ok(guard) = inner.buffer.read()
                        && let Some(data) = guard.as_ref()
                    {
                        inputs.push((data.clone(), inner.shape.clone()));
                    }
                }
                TensorOp::Buffer { .. } => {
                    // Named buffer - also get data if available
                    if let Ok(guard) = inner.buffer.read()
                        && let Some(data) = guard.as_ref()
                    {
                        inputs.push((data.clone(), inner.shape.clone()));
                    }
                }
                // Compute operations - recurse into inputs
                TensorOp::Compute { .. } => {
                    for input in inner.op.inputs() {
                        // Skip const inputs as the lowerer embeds them directly
                        if !matches!(&input.inner.op, TensorOp::Const(_)) {
                            collect_recursive(&input.inner, visited, inputs);
                        }
                    }
                }
                // Other operations - recurse into inputs
                _ => {
                    for input in inner.op.inputs() {
                        collect_recursive(&input.inner, visited, inputs);
                    }
                }
            }
        }

        let mut visited = HashSet::new();
        let mut inputs = Vec::new();
        collect_recursive(&self.inner, &mut visited, &mut inputs);
        inputs
    }

    /// Internal: Execute realize on Metal device
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn realize_metal(&self) -> Result<Tensor<D>, ForwardError> {
        use crate::backend::global::get_default_device;
        use crate::backend::metal::{MetalBuffer, MetalCompiler, MetalDevice};
        use crate::backend::{BufferSignature, KernelSignature, Pipeline};
        use crate::renderer::MetalRenderer;
        use crate::tensor::lowerer::TensorLowerer;

        // Get the Metal device
        let device: std::sync::Arc<MetalDevice> = get_default_device::<MetalDevice>()
            .ok_or_else(|| ForwardError::DeviceUnavailable("Metal device not found".to_string()))?;

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
                let shape_expr: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
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

        let compiled = pipeline
            .compile_ast(ast, signature)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

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

        // Read back result
        let result: Vec<f32> = output_buffer
            .read_vec()
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to read result: {}", e)))?;

        // Create new tensor with executed buffer
        let inner = TensorInner {
            op: TensorOp::Executed,
            view: self.inner.view.clone(),
            shape: self.inner.shape.clone(),
            dtype: self.inner.dtype.clone(),
            name: self.inner.name.clone(),
            autograd: None,
            buffer: RwLock::new(Some(result)),
        };

        Ok(Tensor {
            inner: Arc::new(inner),
            _dim: PhantomData,
        })
    }

    /// Internal: Execute realize on OpenCL device
    #[cfg(feature = "opencl")]
    fn realize_opencl(&self) -> Result<Tensor<D>, ForwardError> {
        use crate::backend::global::get_default_device;
        use crate::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice};
        use crate::backend::{BufferSignature, KernelSignature, Pipeline};
        use crate::renderer::OpenCLRenderer;
        use crate::tensor::lowerer::TensorLowerer;

        // Get the OpenCL device
        let device: std::sync::Arc<OpenCLDevice> = get_default_device::<OpenCLDevice>()
            .ok_or_else(|| {
                ForwardError::DeviceUnavailable("OpenCL device not found".to_string())
            })?;

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
                let shape_expr: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
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

        let compiled = pipeline
            .compile_ast(ast, signature)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

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

        // Read back result
        let result: Vec<f32> = output_buffer
            .read_vec()
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to read result: {}", e)))?;

        // Create new tensor with executed buffer
        let inner = TensorInner {
            op: TensorOp::Executed,
            view: self.inner.view.clone(),
            shape: self.inner.shape.clone(),
            dtype: self.inner.dtype.clone(),
            name: self.inner.name.clone(),
            autograd: None,
            buffer: RwLock::new(Some(result)),
        };

        Ok(Tensor {
            inner: Arc::new(inner),
            _dim: PhantomData,
        })
    }

    /// Create an executed tensor from raw data
    ///
    /// This creates a tensor with the buffer already populated,
    /// bypassing the computation graph.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Tensor<DimDyn> {
        let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::contiguous(shape_exprs);
        let inner = TensorInner {
            op: TensorOp::Executed,
            view,
            shape,
            dtype: DType::F32,
            name: None,
            autograd: None,
            buffer: RwLock::new(Some(data)),
        };

        Tensor {
            inner: Arc::new(inner),
            _dim: PhantomData,
        }
    }

    /// Execute the lazy computation graph on the default device (legacy)
    ///
    /// This method:
    /// 1. Gets the global default device
    /// 2. Builds a Graph from the internal GraphNode
    /// 3. Compiles and optimizes the graph using Pipeline
    /// 4. Executes the kernel on the device
    /// 5. Stores the result in cached_data
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
    /// // Create and compute tensor
    /// let a = Tensor::<Dim2>::full([3, 4], 1.0);
    /// let b = Tensor::<Dim2>::full([3, 4], 2.0);
    /// let c = &a + &b;
    ///
    /// // Execute computation
    /// c.forward()?;
    ///
    /// // Get result
    /// let data = c.data().unwrap();
    /// ```
    pub fn forward(&self) -> Result<(), ForwardError> {
        // Call realize() to execute and get result
        let result = self.realize()?;

        // Copy the result buffer back to self
        if let Some(result_data) = result.data()
            && let Ok(mut guard) = self.inner.buffer.write() {
                *guard = Some(result_data);
            }

        Ok(())
    }

    /// Get the computed data from the tensor
    ///
    /// Returns the data if the tensor has been executed (via contiguous() or forward()).
    ///
    /// Returns None if the tensor has not been executed yet.
    pub fn data(&self) -> Option<Vec<f32>> {
        // Check the buffer field in TensorInner
        if let Ok(guard) = self.inner.buffer.read() {
            return guard.clone();
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
    /// let t = Tensor::<Dim2>::full([2, 3], 1.0);
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
    /// dimension type. This is type-safe: Tensor<Dim2> returns Array<f32, Ix2>, etc.
    ///
    /// Returns None if the tensor has not been executed yet.
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::Ix2;
    ///
    /// let t = Tensor::<Dim2>::full([2, 3], 1.0);
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

impl Tensor<DimDyn> {
    /// Create a tensor from an ndarray with any dimensions
    ///
    /// This creates a DimDyn tensor with the buffer already populated,
    /// bypassing the computation graph.
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::array;
    /// let arr = array![[1.0, 2.0], [3.0, 4.0]];
    /// let t = Tensor::<DimDyn>::from_ndarray(&arr);
    /// assert_eq!(t.shape(), &[2, 2]);
    /// ```
    pub fn from_ndarray<ND: NdDimension>(array: &Array<f32, ND>) -> Tensor<DimDyn> {
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
        impl Tensor<$dim> {
            /// Create a tensor from an ndarray with type-safe dimensions
            ///
            /// This creates a tensor with the buffer already populated,
            /// bypassing the computation graph. The ndarray dimension type
            /// must match the tensor dimension type.
            pub fn from_ndarray(array: &Array<f32, $ix>) -> Tensor<$dim> {
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
                let inner = TensorInner {
                    op: TensorOp::Executed,
                    view,
                    shape,
                    dtype: DType::F32,
                    name: None,
                    autograd: None,
                    buffer: RwLock::new(Some(data)),
                };

                Tensor {
                    inner: Arc::new(inner),
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
        let tensor = Tensor::<Dim1>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.data(), Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_from_ndarray_dim2_type_safe() {
        let arr: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let tensor = Tensor::<Dim2>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_from_ndarray_dim3_type_safe() {
        let arr: Array3<f32> = Array3::zeros((2, 3, 4));
        let tensor = Tensor::<Dim3>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_to_ndarray_dim2_type_safe() {
        let arr: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let tensor = Tensor::<Dim2>::from_ndarray(&arr);

        // Type-safe: returns Array2<f32> automatically
        let recovered: Array2<f32> = tensor.to_ndarray().unwrap();
        assert_eq!(recovered.shape(), &[2, 3]);
        assert_eq!(recovered[[0, 0]], 1.0);
        assert_eq!(recovered[[1, 2]], 6.0);
    }

    #[test]
    fn test_roundtrip_dim2_type_safe() {
        let original: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Convert to Tensor<Dim2>
        let tensor = Tensor::<Dim2>::from_ndarray(&original);

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
        let tensor = Tensor::<DimDyn>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_to_ndarray_dimdyn() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<DimDyn>::from_data(data.clone(), vec![2, 3]);

        // DimDyn returns ArrayD (IxDyn)
        let arr = tensor.to_ndarray().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
    }

    #[test]
    fn test_to_ndarray_dyn_explicit() {
        let arr: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let tensor = Tensor::<Dim2>::from_ndarray(&arr);

        // Can also get dynamic array from static tensor
        let dyn_arr = tensor.to_ndarray_dyn().unwrap();
        assert_eq!(dyn_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_to_ndarray_not_executed() {
        // Create a tensor that hasn't been executed
        let tensor = Tensor::<Dim2>::input("x", [2, 3]);
        // Should return None since it's not executed
        assert!(tensor.to_ndarray().is_none());
    }
}
