//! C backend for Harp (code generation only)
//!
//! This crate provides C code generation for Harp. Unlike the GPU backends
//! (Metal, OpenCL), this backend does not support runtime execution.
//!
//! # Usage
//!
//! ```ignore
//! use harp_backend_c::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Note: C backend cannot be used for runtime execution.
//! // Use it for code generation purposes only.
//! ```

mod device;
pub mod renderer;

pub use device::CDevice;
pub use renderer::{CCode, CRenderer};

// Re-export renderer types for convenience
pub use harp_core::backend::renderer::OptimizationLevel;

use harp_core::backend::device::{BackendRegistry, DeviceError};
use harp_core::backend::global::DeviceKind;
use harp_core::backend::traits::Device;
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// C backend registry implementation
struct CBackendRegistry;

impl BackendRegistry for CBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::C
    }

    fn name(&self) -> &str {
        "C (code generation only)"
    }

    fn is_available(&self) -> bool {
        CDevice::is_available()
    }

    fn create_device(&self, _index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        Ok(Arc::new(CDevice::new()))
    }

    fn list_devices(&self) -> Vec<String> {
        vec!["C Code Generator".to_string()]
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the C backend
///
/// This function registers the C backend with harp-core, making it
/// available for device selection via `HarpDevice::c()`.
///
/// Note: The C backend does NOT support runtime execution. It is only
/// useful for code generation. Attempting to realize() a tensor with
/// the C backend will result in an error.
///
/// This should be called once at program startup. When using the `harp` facade
/// crate, this is done automatically via the `ctor` attribute.
pub fn init() {
    // Register the backend
    harp_core::backend::register_backend(Box::new(CBackendRegistry));

    // Note: We don't register a realizer because C backend doesn't support runtime execution
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
