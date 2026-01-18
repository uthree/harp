//! Rust backend for Eclat with CPU runtime execution
//!
//! This crate provides Rust code generation and CPU runtime execution for Eclat.
//! Unlike GPU backends (Metal, OpenCL), this backend compiles Rust code to a
//! shared library (cdylib) and executes it on the CPU.
//!
//! # Usage
//!
//! ```ignore
//! use eclat_backend_rust::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now Rust backend is available for runtime execution
//! use eclat::backend::set_device_str;
//! set_device_str("rust").unwrap();
//! ```

mod buffer;
mod compiler;
mod device;
pub mod kernel;
pub mod renderer;

pub use buffer::RustBuffer;
pub use compiler::RustCompiler;
pub use device::RustDevice;
pub use kernel::RustKernel;
pub use renderer::{RustCode, RustRenderer};

use eclat::backend::Pipeline;
use eclat::backend::device::{BackendRegistry, DeviceError};
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::{Compiler, Device, TypedBuffer};
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// Rust backend registry implementation
struct RustBackendRegistry;

impl BackendRegistry for RustBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Rust
    }

    fn name(&self) -> &str {
        "Rust (CPU)"
    }

    fn is_available(&self) -> bool {
        RustDevice::is_available()
    }

    fn create_device(&self, _index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        Ok(Arc::new(RustDevice::new()))
    }

    fn list_devices(&self) -> Vec<String> {
        vec!["Rust CPU Backend".to_string()]
    }

    fn supports_runtime(&self) -> bool {
        true // Rust backend supports runtime execution
    }

    fn allocate_buffer(
        &self,
        device: &dyn Any,
        shape: Vec<usize>,
        dtype: eclat::ast::DType,
    ) -> Result<Box<dyn eclat::backend::Buffer>, DeviceError> {
        let rust_device = device.downcast_ref::<RustDevice>().ok_or_else(|| {
            DeviceError::InitializationError("Invalid device type: expected RustDevice".to_string())
        })?;

        let buffer = RustBuffer::allocate(rust_device, shape, dtype).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to allocate Rust buffer: {}", e))
        })?;

        Ok(Box::new(buffer))
    }

    fn compile_ast(
        &self,
        device: &dyn Any,
        program: eclat::ast::AstNode,
        signature: eclat::backend::KernelSignature,
    ) -> Result<Box<dyn eclat::backend::Kernel>, DeviceError> {
        let rust_device = device.downcast_ref::<RustDevice>().ok_or_else(|| {
            DeviceError::InitializationError("Invalid device type: expected RustDevice".to_string())
        })?;

        let renderer = RustRenderer::new();
        let compiler = RustCompiler::new();
        let mut pipeline = Pipeline::new(renderer, compiler, rust_device.clone());

        let cache_entry = pipeline.compile_ast(program, signature).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to compile AST: {}", e))
        })?;

        Ok(cache_entry.kernel)
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the Rust backend
///
/// This function registers the Rust backend with eclat-core, making it
/// available for device selection via `EclatDevice::new("rust")`.
///
/// The Rust backend compiles generated Rust code to a shared library (cdylib)
/// and executes it on the CPU. This requires rustc to be available on the system.
///
/// This is called automatically at program startup via the `ctor` attribute
/// when this crate is linked.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(RustBackendRegistry));
    log::info!("Rust backend initialized");
}

/// Automatic initialization at program startup
#[ctor::ctor]
fn auto_init() {
    init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        // Just ensure init doesn't panic
        init();
    }

    #[test]
    fn test_rust_device_available() {
        let available = RustDevice::is_available();
        println!("Rust backend available: {}", available);
    }
}
