//! OpenCL backend for Harp
//!
//! This crate provides native GPU execution using the `ocl` crate.
//!
//! # Usage
//!
//! ```ignore
//! use eclat_backend_opencl::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now OpenCL is available as a device
//! use eclat::backend::HarpDevice;
//! let device = HarpDevice::opencl(0).unwrap();
//! device.set_as_default();
//! ```

mod buffer;
mod compiler;
mod device;
mod kernel;
pub mod renderer;

pub use buffer::OpenCLBuffer;
pub use compiler::OpenCLCompiler;
pub use device::{OpenCLDevice, OpenCLError};
pub use kernel::OpenCLKernel;

// Re-export renderer types for convenience
pub use eclat::backend::renderer::OptimizationLevel;
pub use renderer::{OpenCLCode, OpenCLRenderer};

use eclat::backend::device::{BackendRegistry, DeviceError};
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::Device;
use ocl::{Device as OclDevice, Platform};
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// OpenCL backend registry implementation
struct OpenCLBackendRegistry;

impl BackendRegistry for OpenCLBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::OpenCL
    }

    fn name(&self) -> &str {
        "OpenCL"
    }

    fn is_available(&self) -> bool {
        OpenCLDevice::is_available()
    }

    fn create_device(&self, index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        let device = OpenCLDevice::with_device(index).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to create OpenCL device: {}", e))
        })?;
        Ok(Arc::new(device))
    }

    fn list_devices(&self) -> Vec<String> {
        let mut devices = Vec::new();
        for platform in Platform::list() {
            if let Ok(ocl_devices) = OclDevice::list_all(platform) {
                for ocl_device in ocl_devices {
                    if let Ok(name) = ocl_device.name() {
                        devices.push(name);
                    }
                }
            }
        }
        devices
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the OpenCL backend
///
/// This function registers the OpenCL backend with eclat-core, making it
/// available for device selection via `HarpDevice::auto()` or `HarpDevice::opencl()`.
///
/// This should be called once at program startup. When using the `eclat` facade
/// crate, this is done automatically via the `ctor` attribute.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(OpenCLBackendRegistry));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        // Just ensure init doesn't panic
        init();
    }
}
