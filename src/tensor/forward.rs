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

use super::{DimDyn, Dimension, Tensor};
use crate::backend::global::{DeviceKind, get_default_device_kind};
use crate::graph::DType;
use std::fmt;
use std::marker::PhantomData;

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
        use crate::backend::Pipeline;
        use crate::backend::global::get_default_device;
        use crate::backend::metal::{MetalBuffer, MetalCompiler, MetalDevice};
        use crate::graph::Graph;
        use crate::renderer::MetalRenderer;
        use std::cell::RefCell;

        // Get the Metal device
        let device: std::sync::Arc<MetalDevice> = get_default_device::<MetalDevice>()
            .ok_or_else(|| ForwardError::DeviceUnavailable("Metal device not found".to_string()))?;

        // Build graph from tensor's node
        let mut graph = Graph::new();
        graph.output("output".to_string(), self.node.clone());

        // Create pipeline and compile
        let renderer = MetalRenderer::default();
        let compiler = MetalCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, device.as_ref().clone());

        let compiled = pipeline
            .compile_graph(graph)
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
        Ok(Tensor {
            node: self.node.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            autograd: self.autograd.clone(),
            buffer: RefCell::new(Some(result)),
            _dim: PhantomData,
        })
    }

    /// Internal: Execute realize on OpenCL device
    #[cfg(feature = "opencl")]
    fn realize_opencl(&self) -> Result<Tensor<D>, ForwardError> {
        use crate::backend::Pipeline;
        use crate::backend::global::get_default_device;
        use crate::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice};
        use crate::graph::Graph;
        use crate::renderer::OpenCLRenderer;
        use std::cell::RefCell;

        // Get the OpenCL device
        let device: std::sync::Arc<OpenCLDevice> = get_default_device::<OpenCLDevice>()
            .ok_or_else(|| {
                ForwardError::DeviceUnavailable("OpenCL device not found".to_string())
            })?;

        // Build graph from tensor's node
        let mut graph = Graph::new();
        graph.output("output".to_string(), self.node.clone());

        // Create pipeline and compile
        let renderer = OpenCLRenderer::default();
        let compiler = OpenCLCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, compiler, device.as_ref().clone());

        let compiled = pipeline
            .compile_graph(graph)
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
        Ok(Tensor {
            node: self.node.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            autograd: self.autograd.clone(),
            buffer: RefCell::new(Some(result)),
            _dim: PhantomData,
        })
    }

    /// Create an executed tensor from raw data
    ///
    /// This creates a tensor with the buffer already populated,
    /// bypassing the computation graph.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Tensor<DimDyn> {
        use crate::ast::Literal;
        use crate::graph::shape::View;
        use crate::graph::{GraphNode, GraphOp};
        use std::cell::RefCell;

        let view = View::contiguous(shape.iter().map(|&s| s as isize).collect::<Vec<_>>());
        let node = GraphNode::new(
            DType::F32,
            GraphOp::ConstFill(Literal::F32(0.0)), // placeholder
            vec![],
            view,
        );

        Tensor {
            node,
            shape,
            dtype: DType::F32,
            autograd: None,
            buffer: RefCell::new(Some(data)),
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
    /// Checks both the buffer field (new API) and autograd.cached_data (legacy).
    ///
    /// Returns None if the tensor has not been executed yet.
    pub fn data(&self) -> Option<Vec<f32>> {
        // First check the buffer field (new API)
        if let Some(data) = self.buffer.borrow().clone() {
            return Some(data);
        }
        // Fall back to autograd.cached_data (legacy)
        self.autograd
            .as_ref()
            .and_then(|ag| ag.cached_data.borrow().clone())
    }

    /// Internal: Execute forward on Metal device
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn forward_metal(&self) -> Result<(), ForwardError> {
        use crate::backend::Pipeline;
        use crate::backend::global::get_default_device;
        use crate::backend::metal::{MetalBuffer, MetalCompiler, MetalDevice, MetalKernel};
        use crate::graph::Graph;
        use crate::renderer::MetalRenderer;

        // Get the Metal device
        let device: std::sync::Arc<MetalDevice> = get_default_device::<MetalDevice>()
            .ok_or_else(|| ForwardError::DeviceUnavailable("Metal device not found".to_string()))?;

        // Build graph from tensor's node
        let mut graph = Graph::new();
        // TODO: Properly trace the graph from this tensor's node
        // For now, this is a placeholder - we need to implement graph building from GraphNode
        graph.output("output".to_string(), self.node.clone());

        // Create pipeline and compile
        let renderer = MetalRenderer::default();
        let compiler = MetalCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, device.as_ref().clone(), compiler);

        let compiled = pipeline
            .compile(graph)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

        // Allocate buffers and execute
        let output_size = self.numel();
        let mut output_buffer = MetalBuffer::new(device.as_ref(), output_size * 4)
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to create buffer: {}", e)))?;

        // TODO: Properly handle input buffers
        // For now, we assume no inputs (constant tensors only)
        let inputs: Vec<&MetalBuffer> = vec![];
        let mut outputs: Vec<&mut MetalBuffer> = vec![&mut output_buffer];

        compiled
            .execute(&inputs, &mut outputs)
            .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

        // Read back result
        let result = output_buffer
            .read()
            .map_err(|e| ForwardError::ExecutionError(format!("Failed to read result: {}", e)))?;

        // Store in cached_data
        if let Some(ref autograd) = self.autograd {
            *autograd.cached_data.borrow_mut() = Some(result);
        }

        Ok(())
    }

    /// Internal: Execute forward on OpenCL device
    #[cfg(feature = "opencl")]
    fn forward_opencl(&self) -> Result<(), ForwardError> {
        use crate::backend::Pipeline;
        use crate::backend::global::get_default_device;
        use crate::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice, OpenCLKernel};
        use crate::graph::Graph;
        use crate::renderer::OpenCLRenderer;

        // Get the OpenCL device
        let device: std::sync::Arc<OpenCLDevice> = get_default_device::<OpenCLDevice>()
            .ok_or_else(|| {
                ForwardError::DeviceUnavailable("OpenCL device not found".to_string())
            })?;

        // Build graph from tensor's node
        let mut graph = Graph::new();
        // TODO: Properly trace the graph from this tensor's node
        graph.output("output".to_string(), self.node.clone());

        // Create pipeline and compile
        let renderer = OpenCLRenderer::default();
        let compiler = OpenCLCompiler::new(device.as_ref().clone());
        let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
            Pipeline::new(renderer, device.as_ref().clone(), compiler);

        let compiled = pipeline
            .compile(graph)
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

        // Store in cached_data
        if let Some(ref autograd) = self.autograd {
            *autograd.cached_data.borrow_mut() = Some(result);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_forward_error_display() {
        let err = ForwardError::NoDefaultDevice;
        assert!(format!("{}", err).contains("No default device"));

        let err = ForwardError::DeviceUnavailable("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = ForwardError::AlreadyExecuted;
        assert!(format!("{}", err).contains("already executed"));
    }

    #[test]
    fn test_forward_no_device() {
        use crate::backend::global::clear_default_device;

        // Clear any existing device
        clear_default_device();

        let t = Tensor::<Dim2>::full([2, 3], 1.0);
        let result = t.forward();
        assert!(matches!(result, Err(ForwardError::NoDefaultDevice)));
    }

    #[test]
    fn test_realize_no_device() {
        use crate::backend::global::clear_default_device;

        // Clear any existing device
        clear_default_device();

        let t = Tensor::<Dim2>::full([2, 3], 1.0);
        let result = t.realize();
        assert!(matches!(result, Err(ForwardError::NoDefaultDevice)));
    }

    #[test]
    fn test_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let t = Tensor::<DimDyn>::from_data(data.clone(), shape.clone());

        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.is_executed());
        assert_eq!(t.data(), Some(data));
    }

    #[test]
    fn test_data_from_buffer() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::<DimDyn>::from_data(data.clone(), vec![2, 2]);

        // Should return data from buffer
        assert_eq!(t.data(), Some(data));
    }

    #[test]
    fn test_is_executed() {
        // Tensor without data is not executed
        let t1 = Tensor::<Dim2>::full([2, 3], 1.0);
        assert!(!t1.is_executed());

        // Tensor with data is executed
        let t2 = Tensor::<DimDyn>::from_data(vec![1.0, 2.0], vec![2]);
        assert!(t2.is_executed());
    }
}
