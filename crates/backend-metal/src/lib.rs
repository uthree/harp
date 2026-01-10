//! Metal backend for Harp (macOS only)
//!
//! This crate provides native GPU execution using Apple's Metal API.
//!
//! # Usage
//!
//! ```ignore
//! use eclat_backend_metal::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now Metal is available as a device
//! use eclat::backend::EclatDevice;
//! let device = EclatDevice::metal(0).unwrap();
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
pub use eclat::backend::renderer::OptimizationLevel;
pub use renderer::{MetalCode, MetalRenderer};

use eclat::backend::device::{BackendRegistry, DeviceError};
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::Device;
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

    fn supports_runtime(&self) -> bool {
        true // Metal supports runtime execution
    }

    fn allocate_buffer(
        &self,
        device: &dyn Any,
        shape: Vec<usize>,
        dtype: eclat::ast::DType,
    ) -> Result<Box<dyn eclat::backend::Buffer>, DeviceError> {
        use eclat::backend::traits::TypedBuffer;

        let metal_device = device.downcast_ref::<MetalDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected MetalDevice".to_string(),
            )
        })?;

        let buffer = MetalBuffer::allocate(metal_device, shape, dtype).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to allocate Metal buffer: {}", e))
        })?;

        Ok(Box::new(buffer))
    }

    fn compile_ast(
        &self,
        device: &dyn Any,
        program: eclat::ast::AstNode,
        signature: eclat::backend::KernelSignature,
    ) -> Result<Box<dyn eclat::backend::Kernel>, DeviceError> {
        use eclat::backend::Pipeline;

        let metal_device = device.downcast_ref::<MetalDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected MetalDevice".to_string(),
            )
        })?;

        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler;
        let mut pipeline = Pipeline::new(renderer, compiler, metal_device.clone());

        let cache_entry = pipeline.compile_ast(program, signature).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to compile AST: {}", e))
        })?;

        Ok(cache_entry.kernel)
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the Metal backend
///
/// This function registers the Metal backend with eclat-core, making it
/// available for device selection via `EclatDevice::auto()` or `EclatDevice::metal()`.
///
/// This should be called once at program startup. When using the `eclat` facade
/// crate, this is done automatically via the `ctor` attribute.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(MetalBackendRegistry));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        // Just ensure init doesn't panic
        init();
    }

    // ========================================================================
    // Integration Tests for Tensor API + GPU Execution
    // ========================================================================

    mod integration {
        use super::*;
        use eclat::ast::DType;
        use eclat::backend::{clear_default_device, set_device_str};
        use eclat::tensor::Tensor;
        use eclat::tensor::dim::{D1, D2};

        fn setup_metal() -> bool {
            // Initialize Metal backend
            init();
            clear_default_device();
            set_device_str("metal").is_ok()
        }

        #[test]
        fn test_tensor_set_data_to_vec() {
            if !setup_metal() {
                println!("Metal not available, skipping test");
                return;
            }

            let x: Tensor<D1, f32> = Tensor::input([4]);
            let input_data = [1.0f32, 2.0, 3.0, 4.0];

            x.set_data(&input_data).expect("set_data failed");
            assert!(x.is_realized());

            let output = x.to_vec().expect("to_vec failed");
            assert_eq!(output, input_data.to_vec());
        }

        #[test]
        fn test_realize_add() {
            if !setup_metal() {
                println!("Metal not available, skipping test");
                return;
            }

            let x: Tensor<D1, f32> = Tensor::input([4]);
            let y: Tensor<D1, f32> = Tensor::input([4]);

            x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data x failed");
            y.set_data(&[5.0f32, 6.0, 7.0, 8.0])
                .expect("set_data y failed");

            let z = &x + &y;
            let result = z.realize();
            assert!(result.is_ok(), "realize failed: {:?}", result.err());

            let output = z.to_vec().expect("to_vec failed");
            assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
        }

        #[test]
        fn test_realize_mul() {
            if !setup_metal() {
                println!("Metal not available, skipping test");
                return;
            }

            let x: Tensor<D1, f32> = Tensor::input([4]);
            let y: Tensor<D1, f32> = Tensor::input([4]);

            x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data x failed");
            y.set_data(&[2.0f32, 3.0, 4.0, 5.0])
                .expect("set_data y failed");

            let z = &x * &y;
            z.realize().expect("realize failed");

            let output = z.to_vec().expect("to_vec failed");
            assert_eq!(output, vec![2.0, 6.0, 12.0, 20.0]);
        }

        #[test]
        fn test_realize_add_2d() {
            if !setup_metal() {
                println!("Metal not available, skipping test");
                return;
            }

            let x: Tensor<D2, f32> = Tensor::input([2, 3]);
            let y: Tensor<D2, f32> = Tensor::input([2, 3]);

            x.set_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("set_data x failed");
            y.set_data(&[6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0])
                .expect("set_data y failed");

            let z = &x + &y;
            z.realize().expect("realize failed");

            let output = z.to_vec().expect("to_vec failed");
            assert_eq!(output, vec![7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
        }
    }
}
