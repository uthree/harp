//! OpenCL native device

use harp_core::backend::traits::Device;
use ocl::{Context as OclContext, Device as OclDevice, Platform, Queue};
use std::sync::Arc;

/// Error type for OpenCL native operations
#[derive(Debug, Clone)]
pub struct OpenCLError(String);

impl std::fmt::Display for OpenCLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenCL error: {}", self.0)
    }
}

impl std::error::Error for OpenCLError {}

impl From<ocl::Error> for OpenCLError {
    fn from(e: ocl::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<String> for OpenCLError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for OpenCLError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// OpenCL native device
///
/// Holds the OpenCL platform, device, context, and command queue.
#[derive(Clone)]
pub struct OpenCLDevice {
    pub(crate) platform: Platform,
    pub(crate) device: OclDevice,
    pub(crate) context: Arc<OclContext>,
    pub(crate) queue: Arc<Queue>,
}

impl Device for OpenCLDevice {
    fn is_available() -> bool {
        !Platform::list().is_empty()
    }
}

impl OpenCLDevice {
    /// Create a new context using the default device
    pub fn new() -> Result<Self, OpenCLError> {
        Self::with_device(0)
    }

    /// Create a new context for a specific device index
    pub fn with_device(device_index: usize) -> Result<Self, OpenCLError> {
        // Get default platform
        let platform = Platform::default();

        // Get all devices
        let devices = OclDevice::list_all(platform)?;
        if devices.is_empty() {
            return Err("No OpenCL devices found".into());
        }

        // Select the requested device
        let device = devices.get(device_index).cloned().ok_or_else(|| {
            format!(
                "Device index {} out of range (available: {})",
                device_index,
                devices.len()
            )
        })?;

        // Create context
        let context = OclContext::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        // Create command queue
        let queue = Queue::new(&context, device, None)?;

        Ok(Self {
            platform,
            device,
            context: Arc::new(context),
            queue: Arc::new(queue),
        })
    }

    /// Get the device name
    pub fn device_name(&self) -> String {
        self.device.name().unwrap_or_else(|_| "Unknown".to_string())
    }

    /// Get the OpenCL platform
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Get the OpenCL context
    pub fn ocl_context(&self) -> &OclContext {
        &self.context
    }

    /// Get the OpenCL command queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get the OpenCL device
    pub fn ocl_device(&self) -> OclDevice {
        self.device
    }

    /// List all available devices
    pub fn list_devices() -> Result<Vec<String>, OpenCLError> {
        let platform = Platform::default();
        let devices = OclDevice::list_all(platform)?;

        devices
            .iter()
            .map(|d| d.name().map_err(OpenCLError::from))
            .collect()
    }
}

// Safety: OpenCL context and queue are thread-safe
unsafe impl Send for OpenCLDevice {}
unsafe impl Sync for OpenCLDevice {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OpenCLBuffer, OpenCLCompiler};
    use harp_core::ast::DType;
    use harp_core::backend::traits::{Buffer, Compiler, Device, KernelConfig};

    #[test]
    fn test_opencl_is_available() {
        // This test just checks if OpenCL is available
        let available = OpenCLDevice::is_available();
        println!("OpenCL available: {}", available);
    }

    #[test]
    fn test_opencl_context_creation() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new();
        assert!(
            context.is_ok(),
            "Failed to create context: {:?}",
            context.err()
        );

        let context = context.unwrap();
        println!("Device name: {}", context.device_name());
    }

    #[test]
    fn test_opencl_buffer_allocation() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new().unwrap();

        // Allocate a buffer
        let shape = vec![4, 4];
        let buffer = OpenCLBuffer::allocate(&context, shape.clone(), DType::F32);
        assert!(
            buffer.is_ok(),
            "Failed to allocate buffer: {:?}",
            buffer.err()
        );

        let buffer = buffer.unwrap();
        assert_eq!(buffer.shape(), &shape);
        assert_eq!(buffer.dtype(), DType::F32);
        assert_eq!(buffer.byte_len(), 4 * 4 * 4); // 16 floats * 4 bytes
    }

    #[test]
    fn test_opencl_buffer_data_transfer() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new().unwrap();

        // Create a buffer and write data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = OpenCLBuffer::from_vec(&context, vec![4], DType::F32, &data).unwrap();

        // Read data back
        let result: Vec<f32> = buffer.read_vec().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_opencl_simple_kernel() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new().unwrap();
        let compiler = OpenCLCompiler::new();

        // Simple kernel that adds two arrays
        let source = r#"
            __kernel void add(__global float* a, __global float* b, __global float* c) {
                int i = get_global_id(0);
                c[i] = a[i] + b[i];
            }
        "#;

        let config = KernelConfig::new("add").with_global_work_size([4, 1, 1]);

        let kernel = compiler.compile(&context, source, config);
        assert!(
            kernel.is_ok(),
            "Failed to compile kernel: {:?}",
            kernel.err()
        );

        // Create input/output buffers
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let a_buffer = OpenCLBuffer::from_vec(&context, vec![4], DType::F32, &a_data).unwrap();
        let b_buffer = OpenCLBuffer::from_vec(&context, vec![4], DType::F32, &b_data).unwrap();
        let c_buffer = OpenCLBuffer::allocate(&context, vec![4], DType::F32).unwrap();

        // Execute kernel
        let kernel = kernel.unwrap();
        let result = kernel.execute_with_buffers(&[&a_buffer, &b_buffer, &c_buffer]);
        assert!(
            result.is_ok(),
            "Failed to execute kernel: {:?}",
            result.err()
        );

        // Read result
        let c_result: Vec<f32> = c_buffer.read_vec().unwrap();
        assert_eq!(c_result, vec![6.0, 8.0, 10.0, 12.0]);
    }
}
