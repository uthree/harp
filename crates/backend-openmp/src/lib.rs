//! OpenMP backend for Eclat with CPU parallel execution
//!
//! This crate provides OpenMP-based parallel code generation and CPU runtime execution.
//! It extends the C backend with OpenMP pragmas for parallel loop execution.
//!
//! # Usage
//!
//! ```ignore
//! use eclat_backend_openmp::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now OpenMP backend is available for runtime execution
//! use eclat::backend::set_device_str;
//! set_device_str("openmp").unwrap();
//! ```

mod buffer;
mod compiler;
mod device;
mod kernel;
pub mod renderer;

pub use buffer::OpenMPBuffer;
pub use compiler::OpenMPCompiler;
pub use device::OpenMPDevice;
pub use kernel::OpenMPKernel;
pub use renderer::OpenMPRenderer;

// Re-export renderer types for convenience
pub use eclat::backend::renderer::OptimizationLevel;

use eclat::backend::device::{BackendRegistry, DeviceError};
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::{Compiler, Device, TypedBuffer};
use eclat::backend::Pipeline;
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// OpenMP backend registry implementation
struct OpenMPBackendRegistry;

impl BackendRegistry for OpenMPBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::OpenMP
    }

    fn name(&self) -> &str {
        "OpenMP (CPU Parallel)"
    }

    fn is_available(&self) -> bool {
        OpenMPDevice::is_available()
    }

    fn create_device(&self, _index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        Ok(Arc::new(OpenMPDevice::new()))
    }

    fn list_devices(&self) -> Vec<String> {
        vec!["OpenMP CPU Backend".to_string()]
    }

    fn supports_runtime(&self) -> bool {
        true
    }

    fn allocate_buffer(
        &self,
        device: &dyn Any,
        shape: Vec<usize>,
        dtype: eclat::ast::DType,
    ) -> Result<Box<dyn eclat::backend::Buffer>, DeviceError> {
        let openmp_device = device.downcast_ref::<OpenMPDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected OpenMPDevice".to_string(),
            )
        })?;

        let buffer = OpenMPBuffer::allocate(openmp_device, shape, dtype).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to allocate OpenMP buffer: {}", e))
        })?;

        Ok(Box::new(buffer))
    }

    fn compile_ast(
        &self,
        device: &dyn Any,
        program: eclat::ast::AstNode,
        signature: eclat::backend::KernelSignature,
    ) -> Result<Box<dyn eclat::backend::Kernel>, DeviceError> {
        let openmp_device = device.downcast_ref::<OpenMPDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected OpenMPDevice".to_string(),
            )
        })?;

        let renderer = OpenMPRenderer::new();
        let compiler = OpenMPCompiler::new();
        let mut pipeline = Pipeline::new(renderer, compiler, openmp_device.clone());

        let cache_entry = pipeline.compile_ast(program, signature).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to compile AST: {}", e))
        })?;

        Ok(cache_entry.kernel)
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the OpenMP backend
///
/// This function registers the OpenMP backend with eclat-core, making it
/// available for device selection via `EclatDevice::new("openmp")`.
///
/// The OpenMP backend compiles generated C code with OpenMP pragmas to a
/// shared library and executes it on the CPU with parallel loop execution.
/// This requires a C compiler with OpenMP support (clang/gcc) to be available.
///
/// This is called automatically at program startup via the `ctor` attribute
/// when this crate is linked.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(OpenMPBackendRegistry));
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

    mod integration {
        use super::*;

        use eclat::backend::{clear_default_device, set_device_str};
        use eclat::tensor::dim::D1;
        use eclat::tensor::Tensor;

        fn setup_openmp() -> bool {
            // Initialize OpenMP backend
            init();
            clear_default_device();
            set_device_str("openmp").is_ok()
        }

        #[test]
        fn test_tensor_set_data_to_vec() {
            if !setup_openmp() {
                println!("OpenMP backend not available, skipping test");
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
            if !setup_openmp() {
                println!("OpenMP backend not available, skipping test");
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
        fn test_realize_add_large() {
            if !setup_openmp() {
                println!("OpenMP backend not available, skipping test");
                return;
            }

            // Larger test to benefit from OpenMP parallelization
            let size = 10000;
            let x: Tensor<D1, f32> = Tensor::input([size]);
            let y: Tensor<D1, f32> = Tensor::input([size]);

            let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let y_data: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

            x.set_data(&x_data).expect("set_data x failed");
            y.set_data(&y_data).expect("set_data y failed");

            let z = &x + &y;
            let result = z.realize();
            assert!(result.is_ok(), "realize failed: {:?}", result.err());

            let output = z.to_vec().expect("to_vec failed");
            // All elements should be equal to size
            assert!(output.iter().all(|&v| (v - size as f32).abs() < 0.001));
        }
    }
}
