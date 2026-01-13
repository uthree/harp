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
//! use eclat::backend::EclatDevice;
//! let device = EclatDevice::opencl(0).unwrap();
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

    fn supports_runtime(&self) -> bool {
        true // OpenCL supports runtime execution
    }

    fn allocate_buffer(
        &self,
        device: &dyn Any,
        shape: Vec<usize>,
        dtype: eclat::ast::DType,
    ) -> Result<Box<dyn eclat::backend::Buffer>, DeviceError> {
        use eclat::backend::traits::TypedBuffer;

        let opencl_device = device.downcast_ref::<OpenCLDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected OpenCLDevice".to_string(),
            )
        })?;

        let buffer = OpenCLBuffer::allocate(opencl_device, shape, dtype).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to allocate OpenCL buffer: {}", e))
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

        let opencl_device = device.downcast_ref::<OpenCLDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected OpenCLDevice".to_string(),
            )
        })?;

        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler;
        let mut pipeline = Pipeline::new(renderer, compiler, opencl_device.clone());

        let cache_entry = pipeline.compile_ast(program, signature).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to compile AST: {}", e))
        })?;

        Ok(cache_entry.kernel)
    }

    fn supports_binary_cache(&self) -> bool {
        true
    }

    fn compile_from_binary(
        &self,
        device: &dyn Any,
        binary: &[u8],
        config: eclat::backend::KernelConfig,
    ) -> Result<Box<dyn eclat::backend::Kernel>, DeviceError> {
        let opencl_device = device.downcast_ref::<OpenCLDevice>().ok_or_else(|| {
            DeviceError::InitializationError(
                "Invalid device type: expected OpenCLDevice".to_string(),
            )
        })?;

        let compiler = OpenCLCompiler;
        let kernel = compiler
            .compile_from_binary(opencl_device, binary, config)
            .map_err(|e| {
                DeviceError::InitializationError(format!("Failed to compile from binary: {}", e))
            })?;

        Ok(Box::new(kernel))
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the OpenCL backend
///
/// This function registers the OpenCL backend with eclat-core, making it
/// available for device selection via `EclatDevice::auto()` or `EclatDevice::opencl()`.
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

    mod integration {
        use super::*;
        use eclat::backend::{clear_default_device, set_device_str};
        use eclat::tensor::Tensor;
        use eclat::tensor::dim::{D1, D2};

        fn setup_opencl() -> bool {
            // Initialize OpenCL backend
            init();
            clear_default_device();
            set_device_str("opencl").is_ok()
        }

        #[test]
        fn test_tensor_set_data_to_vec() {
            if !setup_opencl() {
                println!("OpenCL not available, skipping test");
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
            if !setup_opencl() {
                println!("OpenCL not available, skipping test");
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
            if !setup_opencl() {
                println!("OpenCL not available, skipping test");
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
            if !setup_opencl() {
                println!("OpenCL not available, skipping test");
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

            if !setup_opencl() {
                println!("OpenCL not available, skipping test");
                return;
            }

            // 初期統計を取得
            let initial_stats = get_cache_stats();
            let initial_hits = initial_stats.hits;
            let initial_misses = initial_stats.misses;

            // テンソルを作成
            let x: Tensor<D1, f32> = Tensor::input([4]);
            let y: Tensor<D1, f32> = Tensor::input([4]);

            x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data x failed");
            y.set_data(&[5.0f32, 6.0, 7.0, 8.0])
                .expect("set_data y failed");

            // 1回目の実行: キャッシュミス
            let z1 = &x + &y;
            z1.realize().expect("realize z1 failed");

            let stats_after_first = get_cache_stats();
            assert!(
                stats_after_first.misses > initial_misses,
                "Expected cache miss on first run"
            );

            // バッファをクリアして同じグラフを再実行
            z1.clear_buffer();

            // 2回目の実行: キャッシュヒット
            z1.realize().expect("realize z1 again failed");

            let stats_after_second = get_cache_stats();
            assert!(
                stats_after_second.hits > initial_hits,
                "Expected cache hit on second run"
            );

            // 結果が正しいことも確認
            let output = z1.to_vec().expect("to_vec failed");
            assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
        }

        #[test]
        fn test_binary_cache_support() {
            if !setup_opencl() {
                println!("OpenCL not available, skipping test");
                return;
            }

            let x: Tensor<D1, f32> = Tensor::input([4]);
            let y: Tensor<D1, f32> = Tensor::input([4]);

            x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data x failed");
            y.set_data(&[5.0f32, 6.0, 7.0, 8.0])
                .expect("set_data y failed");

            let z = &x + &y;
            z.realize().expect("realize failed");

            // OpenCLカーネルはバイナリキャッシュをサポートする
            // （これは間接的にテスト - 実際のカーネルにはアクセスできないため）
            let output = z.to_vec().expect("to_vec failed");
            assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
        }
    }
}
