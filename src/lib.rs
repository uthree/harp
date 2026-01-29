//! Eclat: A lightweight tensor library inspired by tinygrad.
//!
//! Eclat provides lazy evaluation, kernel fusion, and multiple backend support
//! for tensor computations.
//!
//! # Example
//!
//! ```rust
//! use eclat::prelude::*;
//!
//! // Initialize the CPU device
//! eclat::init();
//!
//! // Create tensors
//! let x = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
//! let y = Tensor::new([[5.0f32, 6.0], [7.0, 8.0]]);
//!
//! // Lazy operations
//! let z = (&x + &y).sum(None, false);
//!
//! // Realize and get result
//! println!("{}", z.item::<f32>()); // 36.0
//! ```

pub mod autograd;
pub mod device;
pub mod dtype;
pub mod ops;
pub mod runtime;
pub mod schedule;
pub mod shape;
pub mod tensor;
pub mod uop;

pub use autograd::{GradientContext, NoGradGuard};
pub use device::{Buffer, Device, DeviceError};
pub use dtype::{DType, Scalar, ScalarValue};
pub use ops::Ops;
pub use shape::Shape;
pub use tensor::Tensor;
pub use uop::UOp;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::dtype::{DType, Scalar};
    pub use crate::shape::Shape;
    pub use crate::tensor::Tensor;
}

use std::sync::Arc;

/// Initializes the default CPU device.
pub fn init() {
    device::register_device(Arc::new(runtime::CpuDevice::new()));
}

/// Initializes the OpenCL device (requires `opencl` feature).
///
/// Returns an error if no GPU devices are found or OpenCL initialization fails.
#[cfg(feature = "opencl")]
pub fn init_opencl() -> Result<(), DeviceError> {
    let device = runtime::OpenCLDevice::new()
        .map_err(|e| DeviceError::ExecutionFailed(format!("Failed to initialize OpenCL: {}", e)))?;
    device::register_device(Arc::new(device));
    Ok(())
}

/// Initializes the OpenCL device with a specific device index.
///
/// Use this when multiple GPUs are available and you want to select a specific one.
#[cfg(feature = "opencl")]
pub fn init_opencl_with_device(index: usize) -> Result<(), DeviceError> {
    let device = runtime::OpenCLDevice::with_device_index(index).map_err(|e| {
        DeviceError::ExecutionFailed(format!(
            "Failed to initialize OpenCL device {}: {}",
            index, e
        ))
    })?;
    device::register_device(Arc::new(device));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_example() {
        init();

        let x = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
        let y = Tensor::new([[5.0f32, 6.0], [7.0, 8.0]]);
        let z = (&x + &y).sum(None, false);

        assert_eq!(z.item::<f32>(), 36.0);
    }
}
