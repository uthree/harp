//! C backend for Eclat with CPU runtime execution
//!
//! This crate provides C code generation and CPU runtime execution for Eclat.
//! Unlike GPU backends (Metal, OpenCL), this backend compiles C code to a
//! shared library and executes it on the CPU.
//!
//! # Usage
//!
//! ```ignore
//! use eclat_backend_c::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now C backend is available for runtime execution
//! use eclat::backend::set_device_str;
//! set_device_str("c").unwrap();
//! ```

mod buffer;
mod compiler;
mod device;
mod kernel;
pub mod renderer;

pub use buffer::CBuffer;
pub use compiler::CCompiler;
pub use device::CDevice;
pub use kernel::CKernel;
pub use renderer::{CCode, CRenderer};

// Re-export renderer types for convenience
pub use eclat::backend::renderer::OptimizationLevel;

use eclat::backend::Pipeline;
use eclat::backend::device::{BackendRegistry, DeviceError};
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::{Compiler, Device, TypedBuffer};
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
        "C (CPU)"
    }

    fn is_available(&self) -> bool {
        CDevice::is_available()
    }

    fn create_device(&self, _index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        Ok(Arc::new(CDevice::new()))
    }

    fn list_devices(&self) -> Vec<String> {
        vec!["C CPU Backend".to_string()]
    }

    fn supports_runtime(&self) -> bool {
        true // C backend now supports runtime execution
    }

    fn allocate_buffer(
        &self,
        device: &dyn Any,
        shape: Vec<usize>,
        dtype: eclat::ast::DType,
    ) -> Result<Box<dyn eclat::backend::Buffer>, DeviceError> {
        let c_device = device.downcast_ref::<CDevice>().ok_or_else(|| {
            DeviceError::InitializationError("Invalid device type: expected CDevice".to_string())
        })?;

        let buffer = CBuffer::allocate(c_device, shape, dtype).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to allocate C buffer: {}", e))
        })?;

        Ok(Box::new(buffer))
    }

    fn compile_ast(
        &self,
        device: &dyn Any,
        program: eclat::ast::AstNode,
        signature: eclat::backend::KernelSignature,
    ) -> Result<Box<dyn eclat::backend::Kernel>, DeviceError> {
        let c_device = device.downcast_ref::<CDevice>().ok_or_else(|| {
            DeviceError::InitializationError("Invalid device type: expected CDevice".to_string())
        })?;

        let renderer = CRenderer::new();
        let compiler = CCompiler::new();
        let mut pipeline = Pipeline::new(renderer, compiler, c_device.clone());

        let cache_entry = pipeline.compile_ast(program, signature).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to compile AST: {}", e))
        })?;

        Ok(cache_entry.kernel)
    }
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the C backend
///
/// This function registers the C backend with eclat-core, making it
/// available for device selection via `EclatDevice::c()`.
///
/// The C backend compiles generated C code to a shared library and
/// executes it on the CPU. This requires a C compiler (clang/gcc/cl)
/// to be available on the system.
///
/// This is called automatically at program startup via the `ctor` attribute
/// when this crate is linked.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(CBackendRegistry));
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
        use eclat::tensor::Tensor;
        use eclat::tensor::dim::{D1, D2};

        fn setup_c() -> bool {
            // Initialize C backend
            init();
            clear_default_device();
            set_device_str("c").is_ok()
        }

        #[test]
        fn test_tensor_set_data_to_vec() {
            if !setup_c() {
                println!("C backend not available, skipping test");
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
            if !setup_c() {
                println!("C backend not available, skipping test");
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
            if !setup_c() {
                println!("C backend not available, skipping test");
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
            if !setup_c() {
                println!("C backend not available, skipping test");
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

            if !setup_c() {
                println!("C backend not available, skipping test");
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
    }
}
