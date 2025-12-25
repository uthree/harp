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

        // Lower Tensor to AST directly
        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&self.clone().into_dyn());

        // Create signature from tensor metadata
        let output_shape: Vec<Expr> = self.shape().iter().map(|&s| Expr::from(s as i64)).collect();
        let signature = KernelSignature::new(
            vec![], // TODO: Collect input signatures
            vec![BufferSignature::new("output".to_string(), output_shape)],
        );

        // Create pipeline and compile from AST
        let renderer = MetalRenderer::default();
        let compiler = MetalCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, device.as_ref().clone());

        let compiled = pipeline
            .compile_ast(ast, signature)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

        // Allocate buffers and execute
        let output_size = self.numel();
        let mut output_buffer = MetalBuffer::new(device.as_ref(), output_size * 4)
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to create buffer: {}", e)))?;

        // TODO: Properly handle input buffers
        let inputs: Vec<&MetalBuffer> = vec![];
        let mut outputs: Vec<&mut MetalBuffer> = vec![&mut output_buffer];

        compiled
            .execute(&inputs, &mut outputs)
            .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

        // Read back result
        let result = output_buffer
            .read()
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

        // Lower Tensor to AST directly
        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&self.clone().into_dyn());

        // Create signature from tensor metadata
        let output_shape: Vec<Expr> = self.shape().iter().map(|&s| Expr::from(s as i64)).collect();
        let signature = KernelSignature::new(
            vec![], // TODO: Collect input signatures
            vec![BufferSignature::new("output".to_string(), output_shape)],
        );

        // Create pipeline and compile from AST
        let renderer = OpenCLRenderer::default();
        let compiler = OpenCLCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, device.as_ref().clone());

        let compiled = pipeline
            .compile_ast(ast, signature)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

        // Allocate buffers and execute
        let output_size = self.numel();
        let mut output_buffer = OpenCLBuffer::new(device.as_ref(), output_size * 4)
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to create buffer: {}", e)))?;

        // TODO: Properly handle input buffers
        let inputs: Vec<&OpenCLBuffer> = vec![];
        let mut outputs: Vec<&mut OpenCLBuffer> = vec![&mut output_buffer];

        compiled
            .execute(&inputs, &mut outputs)
            .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

        // Read back result
        let result = output_buffer
            .read()
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
        let device_kind = get_default_device_kind();

        match device_kind {
            DeviceKind::None => Err(ForwardError::NoDefaultDevice),

            #[cfg(all(feature = "metal", target_os = "macos"))]
            DeviceKind::Metal => self.forward_metal(),

            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            DeviceKind::Metal => Err(ForwardError::DeviceUnavailable(
                "Metal backend is not available. Enable 'metal' feature on macOS.".to_string(),
            )),

            #[cfg(feature = "opencl")]
            DeviceKind::OpenCL => self.forward_opencl(),

            #[cfg(not(feature = "opencl"))]
            DeviceKind::OpenCL => Err(ForwardError::DeviceUnavailable(
                "OpenCL backend is not available. Enable 'opencl' feature.".to_string(),
            )),
        }
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
    /// let arr = t.to_ndarray().unwrap();
    /// assert_eq!(arr.shape(), &[2, 3]);
    /// ```
    pub fn to_ndarray(&self) -> Option<ArrayD<f32>> {
        let data = self.data()?;
        let shape = IxDyn(self.shape());
        Some(Array::from_shape_vec(shape, data).expect("Shape mismatch in to_ndarray"))
    }

    /// Convert the tensor to an ndarray with static dimensions
    ///
    /// Returns the data as an ndarray with the specified dimension type if executed.
    /// Returns None if the tensor has not been executed yet.
    ///
    /// # Type Parameters
    /// * `ND` - The ndarray dimension type (e.g., `Ix2` for 2D arrays)
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::Ix2;
    /// let t = Tensor::<Dim2>::full([2, 3], 1.0);
    /// t.forward()?;
    /// let arr: ndarray::Array<f32, Ix2> = t.to_ndarray_d().unwrap();
    /// ```
    pub fn to_ndarray_d<ND: NdDimension>(&self) -> Option<Array<f32, ND>> {
        let data = self.data()?;
        let shape =
            ND::from_dimension(&IxDyn(self.shape())).expect("Dimension mismatch in to_ndarray_d");
        Some(Array::from_shape_vec(shape, data).expect("Shape mismatch in to_ndarray_d"))
    }
}

impl Tensor<DimDyn> {
    /// Create a tensor from an ndarray with dynamic dimensions
    ///
    /// This creates a tensor with the buffer already populated,
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

    /// Create a tensor from a dynamic ndarray
    ///
    /// Convenience method for creating a tensor from ArrayD<f32>.
    pub fn from_ndarray_dyn(array: &ArrayD<f32>) -> Tensor<DimDyn> {
        let shape = array.shape().to_vec();
        let data: Vec<f32> = array
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| array.iter().cloned().collect());
        Self::from_data(data, shape)
    }

    /// Internal: Execute forward on Metal device
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn forward_metal(&self) -> Result<(), ForwardError> {
        use crate::backend::global::get_default_device;
        use crate::backend::metal::{MetalBuffer, MetalCompiler, MetalDevice};
        use crate::backend::{BufferSignature, KernelSignature, Pipeline};
        use crate::renderer::MetalRenderer;
        use crate::tensor::lowerer::TensorLowerer;

        // Get the Metal device
        let device: std::sync::Arc<MetalDevice> = get_default_device::<MetalDevice>()
            .ok_or_else(|| ForwardError::DeviceUnavailable("Metal device not found".to_string()))?;

        // Lower Tensor to AST directly
        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&self.clone().into_dyn());

        // Create signature from tensor metadata
        let output_shape: Vec<Expr> = self.shape().iter().map(|&s| Expr::from(s as i64)).collect();
        let signature = KernelSignature::new(
            vec![], // TODO: Collect input signatures
            vec![BufferSignature::new("output".to_string(), output_shape)],
        );

        // Create pipeline and compile from AST
        let renderer = MetalRenderer::default();
        let compiler = MetalCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, device.as_ref().clone());

        let compiled = pipeline
            .compile_ast(ast, signature)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

        // Allocate buffers and execute
        let output_size = self.numel();
        let mut output_buffer = MetalBuffer::new(device.as_ref(), output_size * 4)
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to create buffer: {}", e)))?;

        // TODO: Properly handle input buffers
        let inputs: Vec<&MetalBuffer> = vec![];
        let mut outputs: Vec<&mut MetalBuffer> = vec![&mut output_buffer];

        compiled
            .execute(&inputs, &mut outputs)
            .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

        // Read back result
        let result = output_buffer
            .read()
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to read result: {}", e)))?;

        // Store result in buffer (note: this requires interior mutability)
        if let Ok(mut guard) = self.inner.buffer.write() {
            *guard = Some(result);
        }

        Ok(())
    }

    /// Internal: Execute forward on OpenCL device
    #[cfg(feature = "opencl")]
    fn forward_opencl(&self) -> Result<(), ForwardError> {
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

        // Lower Tensor to AST directly
        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&self.clone().into_dyn());

        // Create signature from tensor metadata
        let output_shape: Vec<Expr> = self.shape().iter().map(|&s| Expr::from(s as i64)).collect();
        let signature = KernelSignature::new(
            vec![], // TODO: Collect input signatures
            vec![BufferSignature::new("output".to_string(), output_shape)],
        );

        // Create pipeline and compile from AST
        let renderer = OpenCLRenderer::default();
        let compiler = OpenCLCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, device.as_ref().clone());

        let compiled = pipeline
            .compile_ast(ast, signature)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

        // Allocate buffers and execute
        let output_size = self.numel();
        let mut output_buffer = OpenCLBuffer::new(device.as_ref(), output_size * 4)
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to create buffer: {}", e)))?;

        // TODO: Properly handle input buffers
        let inputs: Vec<&OpenCLBuffer> = vec![];
        let mut outputs: Vec<&mut OpenCLBuffer> = vec![&mut output_buffer];

        compiled
            .execute(&inputs, &mut outputs)
            .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

        // Read back result
        let result = output_buffer
            .read()
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to read result: {}", e)))?;

        // Store result in buffer
        if let Ok(mut guard) = self.inner.buffer.write() {
            *guard = Some(result);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Ix2, array};

    #[test]
    fn test_from_ndarray_2d() {
        let arr = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::<DimDyn>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_from_ndarray_3d() {
        let arr = ndarray::Array3::<f32>::zeros((2, 3, 4));
        let tensor = Tensor::<DimDyn>::from_ndarray(&arr);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_to_ndarray() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<DimDyn>::from_data(data.clone(), vec![2, 3]);

        let arr = tensor.to_ndarray().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[0, 1]], 2.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_to_ndarray_d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::<DimDyn>::from_data(data, vec![2, 2]);

        let arr: Array2<f32> = tensor.to_ndarray_d::<Ix2>().unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }

    #[test]
    fn test_roundtrip_ndarray() {
        // Create original ndarray
        let original = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Convert to Tensor
        let tensor = Tensor::<DimDyn>::from_ndarray(&original);

        // Convert back to ndarray
        let recovered = tensor.to_ndarray().unwrap();

        // Verify they match
        assert_eq!(original.shape(), recovered.shape());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_from_ndarray_dyn() {
        let arr = ndarray::ArrayD::<f32>::zeros(vec![2, 3, 4]);
        let tensor = Tensor::<DimDyn>::from_ndarray_dyn(&arr);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_to_ndarray_not_executed() {
        use crate::tensor::Dim2;
        // Create a tensor that hasn't been executed
        let tensor = Tensor::<Dim2>::input("x", [2, 3]);
        // Should return None since it's not executed
        assert!(tensor.to_ndarray().is_none());
    }
}
