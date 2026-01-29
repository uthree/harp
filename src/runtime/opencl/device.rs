//! OpenCL device implementation.

use std::sync::{Arc, RwLock};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device as ClDevice, get_all_devices};

use crate::device::{Buffer, BufferMap, Device, DeviceError, Result};
use crate::dtype::DType;
use crate::uop::UOp;

use super::buffer::OpenCLBuffer;
use super::interpreter::OpenCLInterpreter;
use super::kernel::KernelCache;

/// OpenCL device implementation.
pub struct OpenCLDevice {
    context: Context,
    queue: Arc<CommandQueue>,
    device: ClDevice,
    kernel_cache: RwLock<KernelCache>,
}

impl OpenCLDevice {
    /// Creates a new OpenCL device using the first available GPU.
    pub fn new() -> Result<Self> {
        // Get all GPU devices
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU).map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to get GPU devices: {:?}", e))
        })?;

        if device_ids.is_empty() {
            return Err(DeviceError::ExecutionFailed("No GPU devices found".into()));
        }

        let device = ClDevice::new(device_ids[0]);

        // Create context
        let context = Context::from_device(&device).map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to create context: {:?}", e))
        })?;

        // Create command queue
        let queue = CommandQueue::create_default(&context, 0).map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to create queue: {:?}", e))
        })?;

        Ok(OpenCLDevice {
            context,
            queue: Arc::new(queue),
            device,
            kernel_cache: RwLock::new(KernelCache::new()),
        })
    }

    /// Creates an OpenCL device with a specific device index.
    pub fn with_device_index(index: usize) -> Result<Self> {
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU).map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to get GPU devices: {:?}", e))
        })?;

        if index >= device_ids.len() {
            return Err(DeviceError::ExecutionFailed(format!(
                "Device index {} out of range (found {} devices)",
                index,
                device_ids.len()
            )));
        }

        let device = ClDevice::new(device_ids[index]);

        let context = Context::from_device(&device).map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to create context: {:?}", e))
        })?;

        let queue = CommandQueue::create_default(&context, 0).map_err(|e| {
            DeviceError::ExecutionFailed(format!("Failed to create queue: {:?}", e))
        })?;

        Ok(OpenCLDevice {
            context,
            queue: Arc::new(queue),
            device,
            kernel_cache: RwLock::new(KernelCache::new()),
        })
    }

    /// Returns the OpenCL context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the command queue.
    pub fn queue(&self) -> &Arc<CommandQueue> {
        &self.queue
    }

    /// Returns the underlying OpenCL device.
    pub fn cl_device(&self) -> &ClDevice {
        &self.device
    }

    /// Returns the kernel cache.
    pub fn kernel_cache(&self) -> &RwLock<KernelCache> {
        &self.kernel_cache
    }

    /// Returns device info string.
    pub fn device_info(&self) -> String {
        let name = self.device.name().unwrap_or_else(|_| "Unknown".into());
        let vendor = self.device.vendor().unwrap_or_else(|_| "Unknown".into());
        format!("{} ({})", name, vendor)
    }
}

impl Device for OpenCLDevice {
    fn name(&self) -> &str {
        "OPENCL"
    }

    fn alloc(&self, numel: usize, dtype: DType) -> Result<Box<dyn Buffer>> {
        let buffer = OpenCLBuffer::new(&self.context, self.queue.clone(), numel, dtype)?;
        Ok(Box::new(buffer))
    }

    fn realize(&self, uop: &UOp, buffers: &mut BufferMap) -> Result<Arc<dyn Buffer>> {
        let mut interpreter = OpenCLInterpreter::new(self, buffers);
        interpreter.eval(uop)
    }
}

// OpenCL handles thread safety internally
unsafe impl Send for OpenCLDevice {}
unsafe impl Sync for OpenCLDevice {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::ScalarValue;
    use crate::runtime::opencl::buffer::OpenCLBuffer;
    use crate::shape::Shape;

    fn get_test_device() -> Option<OpenCLDevice> {
        OpenCLDevice::new().ok()
    }

    #[test]
    fn test_opencl_device_creation() {
        if let Some(device) = get_test_device() {
            assert_eq!(device.name(), "OPENCL");
            println!("OpenCL device: {}", device.device_info());
        } else {
            println!("No OpenCL GPU device available, skipping test");
        }
    }

    #[test]
    fn test_opencl_buffer() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let buffer = device.alloc(4, DType::Float32).unwrap();
        assert_eq!(buffer.size(), 16); // 4 * 4 bytes
        assert_eq!(buffer.dtype(), DType::Float32);

        // Test host transfer
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect();

        let mut buffer =
            OpenCLBuffer::new(device.context(), device.queue().clone(), 4, DType::Float32).unwrap();
        buffer.copy_from_host(&data);

        let result = buffer.copy_to_host();
        assert_eq!(result, data);
    }

    #[test]
    fn test_opencl_const() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let mut buffers = BufferMap::new();
        let uop = UOp::constant(ScalarValue::Float32(3.14), Shape::new([2, 2]));
        let result = device.realize(&uop, &mut buffers).unwrap();

        let data = result.copy_to_host();
        let values: Vec<f32> = data
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for v in &values {
            assert!((v - 3.14).abs() < 0.0001);
        }
    }

    #[test]
    fn test_opencl_add() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let mut buffers = BufferMap::new();

        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 2]));
        let b = UOp::constant(ScalarValue::Float32(2.0), Shape::new([2, 2]));
        let c = a.add(&b);

        let result = device.realize(&c, &mut buffers).unwrap();
        let data = result.copy_to_host();
        let values: Vec<f32> = data
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for v in &values {
            assert!((v - 3.0).abs() < 0.0001);
        }
    }

    #[test]
    fn test_opencl_unary() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let mut buffers = BufferMap::new();

        let a = UOp::constant(ScalarValue::Float32(2.0), Shape::new([4]));
        let neg = a.neg();

        let result = device.realize(&neg, &mut buffers).unwrap();
        let data = result.copy_to_host();
        let values: Vec<f32> = data
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for v in &values {
            assert!((v - (-2.0)).abs() < 0.0001);
        }
    }

    #[test]
    fn test_opencl_sum() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let mut buffers = BufferMap::new();

        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3]));
        let s = a.sum(None, false);

        let result = device.realize(&s, &mut buffers).unwrap();
        let data = result.copy_to_host();
        let value = f32::from_ne_bytes([data[0], data[1], data[2], data[3]]);

        println!("Sum result: {} (expected 6.0)", value);
        assert!((value - 6.0).abs() < 0.0001, "Got {}, expected 6.0", value);
    }

    #[test]
    fn test_opencl_broadcast_add() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let mut buffers = BufferMap::new();

        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3]));
        let b = UOp::constant(ScalarValue::Float32(2.0), Shape::new([1, 3]));
        let c = a.add(&b);

        let result = device.realize(&c, &mut buffers).unwrap();
        let data = result.copy_to_host();
        let values: Vec<f32> = data
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(values.len(), 6);
        for v in &values {
            assert!((v - 3.0).abs() < 0.0001);
        }
    }

    #[test]
    fn test_opencl_fused_elementwise() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let mut buffers = BufferMap::new();

        // Create: (a + b) * c - should fuse add and mul into one kernel
        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([4]));
        let b = UOp::constant(ScalarValue::Float32(2.0), Shape::new([4]));
        let c = UOp::constant(ScalarValue::Float32(3.0), Shape::new([4]));

        let sum = a.add(&b);
        let product = sum.mul(&c);

        // Test with fusion
        let mut interpreter = OpenCLInterpreter::new(&device, &mut buffers);
        let result = interpreter.eval_with_fusion(&product).unwrap();

        let data = result.copy_to_host();
        let values: Vec<f32> = data
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // (1 + 2) * 3 = 9
        for v in &values {
            assert!(
                (v - 9.0).abs() < 0.0001,
                "Expected 9.0, got {}",
                v
            );
        }
    }

    #[test]
    fn test_opencl_fused_chain() {
        let Some(device) = get_test_device() else {
            println!("No OpenCL GPU device available, skipping test");
            return;
        };

        let mut buffers = BufferMap::new();

        // Create: neg(a + b) - triple chain fusion
        let a = UOp::constant(ScalarValue::Float32(2.0), Shape::new([4]));
        let b = UOp::constant(ScalarValue::Float32(3.0), Shape::new([4]));

        let sum = a.add(&b);
        let neg = sum.neg();

        let mut interpreter = OpenCLInterpreter::new(&device, &mut buffers);
        let result = interpreter.eval_with_fusion(&neg).unwrap();

        let data = result.copy_to_host();
        let values: Vec<f32> = data
            .chunks(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // -(2 + 3) = -5
        for v in &values {
            assert!(
                (v - (-5.0)).abs() < 0.0001,
                "Expected -5.0, got {}",
                v
            );
        }
    }
}
