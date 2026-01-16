//! CUDA backend for Eclat with GPU runtime execution
//!
//! This crate provides CUDA code generation and GPU runtime execution for Eclat.
//! It uses nvcc to compile CUDA kernels and cudarc for runtime execution.
//!
//! # Features
//!
//! - `cuda-runtime`: Enable full CUDA runtime support (requires CUDA SDK)
//!
//! # Usage
//!
//! ```ignore
//! use eclat_backend_cuda::init;
//!
//! // Initialize the backend
//! init();
//!
//! // Now CUDA backend is available for runtime execution
//! use eclat::backend::set_device_str;
//! set_device_str("cuda:0").unwrap();
//! ```

mod buffer;
mod compiler;
mod device;
pub mod kernel;
pub mod renderer;

pub use buffer::CudaBuffer;
pub use compiler::CudaCompiler;
pub use device::CudaDevice;
pub use kernel::CudaKernel;
pub use renderer::{CudaCode, CudaRenderer};

use eclat::backend::device::{BackendRegistry, DeviceError};
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::Compiler;
use eclat::backend::Pipeline;
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// CUDA backend registry implementation
struct CudaBackendRegistry;

impl BackendRegistry for CudaBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn name(&self) -> &str {
        "CUDA (GPU)"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "cuda-runtime")]
        {
            use eclat::backend::traits::Device;
            CudaDevice::is_available()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            false
        }
    }

    fn create_device(&self, index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        let device = CudaDevice::new(index).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to create CUDA device: {}", e))
        })?;
        Ok(Arc::new(device))
    }

    fn list_devices(&self) -> Vec<String> {
        CudaDevice::list_devices()
    }

    fn supports_runtime(&self) -> bool {
        cfg!(feature = "cuda-runtime")
    }

    fn allocate_buffer(
        &self,
        device: &dyn Any,
        shape: Vec<usize>,
        dtype: eclat::ast::DType,
    ) -> Result<Box<dyn eclat::backend::Buffer>, DeviceError> {
        let cuda_device = device.downcast_ref::<CudaDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected CudaDevice".to_string(),
            )
        })?;

        let buffer = CudaBuffer::allocate(cuda_device, shape, dtype).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to allocate CUDA buffer: {}", e))
        })?;

        Ok(Box::new(buffer))
    }

    fn compile_ast(
        &self,
        device: &dyn Any,
        program: eclat::ast::AstNode,
        signature: eclat::backend::KernelSignature,
    ) -> Result<Box<dyn eclat::backend::Kernel>, DeviceError> {
        let cuda_device = device.downcast_ref::<CudaDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected CudaDevice".to_string(),
            )
        })?;

        let renderer = CudaRenderer::new();
        let compiler = CudaCompiler::new();
        let mut pipeline = Pipeline::new(renderer, compiler, cuda_device.clone());

        let cache_entry = pipeline.compile_ast(program, signature).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to compile AST: {}", e))
        })?;

        Ok(cache_entry.kernel)
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the CUDA backend
///
/// This function registers the CUDA backend with eclat-core, making it
/// available for device selection via `EclatDevice::new("cuda:0")`.
///
/// The CUDA backend requires:
/// - NVIDIA GPU with CUDA support
/// - CUDA Toolkit installed (nvcc compiler)
/// - CUDA drivers installed
///
/// Note: If built without the `cuda-runtime` feature, the backend will
/// register but report as unavailable.
///
/// This is called automatically at program startup via the `ctor` attribute
/// when this crate is linked.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(CudaBackendRegistry));
    log::info!("CUDA backend initialized");
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
    fn test_cuda_availability_check() {
        // This test just checks that availability check doesn't panic
        #[cfg(feature = "cuda-runtime")]
        {
            use eclat::backend::traits::Device;
            let available = CudaDevice::is_available();
            println!("CUDA backend available: {}", available);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            println!("CUDA runtime feature not enabled");
        }
    }

    #[test]
    fn test_nvcc_check() {
        let nvcc_available = CudaDevice::is_nvcc_available();
        println!("nvcc available: {}", nvcc_available);
    }

    #[test]
    fn test_renderer_basic() {
        use eclat::ast::DType;

        let renderer = CudaRenderer::new();

        // Test basic dtype rendering
        use eclat::backend::renderer::CLikeRenderer;
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::I32), "int");
        assert_eq!(renderer.render_dtype_backend(&DType::Bool), "bool");

        // Test kernel qualifier
        assert_eq!(renderer.render_function_qualifier(true), "__global__ ");
        assert_eq!(renderer.render_function_qualifier(false), "__device__ ");

        // Test barrier
        assert_eq!(renderer.render_barrier_backend(), "__syncthreads();");
    }

    /// Integration tests that require CUDA runtime
    #[cfg(feature = "cuda-runtime")]
    mod integration {
        use super::*;

        use eclat::backend::traits::Device;
        use eclat::backend::{clear_default_device, set_device_str};
        use eclat::tensor::dim::{D1, D2};
        use eclat::tensor::Tensor;

        fn setup_cuda() -> bool {
            // Initialize CUDA backend
            init();
            clear_default_device();

            // Check if CUDA is actually available
            if !CudaDevice::is_available() {
                return false;
            }

            set_device_str("cuda:0").is_ok()
        }

        #[test]
        fn test_tensor_set_data_to_vec() {
            if !setup_cuda() {
                println!("CUDA backend not available, skipping test");
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
            if !setup_cuda() {
                println!("CUDA backend not available, skipping test");
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
            if !setup_cuda() {
                println!("CUDA backend not available, skipping test");
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
            if !setup_cuda() {
                println!("CUDA backend not available, skipping test");
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

        #[test]
        fn test_kernel_cache_hit() {
            use eclat::backend::get_cache_stats;

            if !setup_cuda() {
                println!("CUDA backend not available, skipping test");
                return;
            }

            // Get initial stats
            let initial_stats = get_cache_stats();
            let initial_hits = initial_stats.hits;
            let initial_misses = initial_stats.misses;

            // Create tensors
            let x: Tensor<D1, f32> = Tensor::input([4]);
            let y: Tensor<D1, f32> = Tensor::input([4]);

            x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data x failed");
            y.set_data(&[5.0f32, 6.0, 7.0, 8.0])
                .expect("set_data y failed");

            // First execution: cache miss
            let z1 = &x + &y;
            z1.realize().expect("realize z1 failed");

            let stats_after_first = get_cache_stats();
            assert!(
                stats_after_first.misses > initial_misses,
                "Expected cache miss on first run"
            );

            // Clear buffer and re-run same graph
            z1.clear_buffer();

            // Second execution: cache hit
            z1.realize().expect("realize z1 again failed");

            let stats_after_second = get_cache_stats();
            assert!(
                stats_after_second.hits > initial_hits,
                "Expected cache hit on second run"
            );

            // Verify result is still correct
            let output = z1.to_vec().expect("to_vec failed");
            assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
        }
    }
}
