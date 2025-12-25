//! Forward execution for Tensor
//!
//! This module provides the forward() implementation which compiles and
//! executes the lazy computation graph on the default device.

use super::{Dimension, Tensor};
use crate::backend::global::{DeviceKind, get_default_device_kind};
use std::fmt;

/// Error type for forward execution
#[derive(Debug)]
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
        }
    }
}

impl std::error::Error for ForwardError {}

impl<D: Dimension> Tensor<D> {
    /// Execute the lazy computation graph on the default device
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

    /// Get the cached data from forward execution
    ///
    /// Returns None if forward() has not been called yet.
    pub fn data(&self) -> Option<Vec<f32>> {
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
}
