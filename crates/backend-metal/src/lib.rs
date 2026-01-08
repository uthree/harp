//! Metal backend for Harp (macOS only)
//!
//! This crate provides native GPU execution using Apple's Metal API.
//!
//! # Usage
//!
//! ```ignore
//! use harp_backend_metal::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now Metal is available as a device
//! use harp::backend::HarpDevice;
//! let device = HarpDevice::metal(0).unwrap();
//! device.set_as_default();
//! ```

#![cfg(target_os = "macos")]

mod buffer;
mod compiler;
mod device;
mod kernel;
pub mod renderer;

pub use buffer::MetalBuffer;
pub use compiler::MetalCompiler;
pub use device::{MetalDevice, MetalError};
pub use kernel::MetalKernel;

// Re-export renderer types for convenience
pub use harp::backend::renderer::OptimizationLevel;
pub use renderer::{MetalCode, MetalRenderer};

use harp::backend::device::{BackendRegistry, DeviceError};
use harp::backend::global::DeviceKind;
use harp::backend::traits::Device;
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// Metal backend registry implementation
struct MetalBackendRegistry;

impl BackendRegistry for MetalBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn name(&self) -> &str {
        "Metal"
    }

    fn is_available(&self) -> bool {
        MetalDevice::is_available()
    }

    fn create_device(&self, index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        let device = MetalDevice::with_device(index).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to create Metal device: {}", e))
        })?;
        Ok(Arc::new(device))
    }

    fn list_devices(&self) -> Vec<String> {
        metal::Device::all()
            .into_iter()
            .map(|d| d.name().to_string())
            .collect()
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the Metal backend
///
/// This function registers the Metal backend with harp-core, making it
/// available for device selection via `HarpDevice::auto()` or `HarpDevice::metal()`.
///
/// This should be called once at program startup. When using the `harp` facade
/// crate, this is done automatically via the `ctor` attribute.
pub fn init() {
    // Register the backend
    harp::backend::register_backend(Box::new(MetalBackendRegistry));
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
