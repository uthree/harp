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
/// This is called automatically at program startup via the `ctor` attribute
/// when this crate is linked.
pub fn init() {
    // Register the backend
    eclat::backend::register_backend(Box::new(MetalBackendRegistry));
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

    // ========================================================================
    // Integration Tests for Tensor API + GPU Execution
    // ========================================================================

    mod integration {
        use super::*;
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

        #[test]
        fn test_autograd_backward_square() {
            // Test: y = x^2, loss = sum(y)
            // Expected gradient: dy/dx = 2x
            if !setup_metal() {
                println!("Metal not available, skipping test");
                return;
            }

            let x: Tensor<D1, f32> = Tensor::input([4]);
            x.requires_grad_(true);
            x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data failed");

            let y = &x * &x; // y = x^2
            let loss = y.sum(0); // scalar

            // Compute gradients
            loss.backward_with_params(&[&x]).expect("backward failed");

            // Get and realize the gradient
            let grad = x.grad().expect("gradient should exist");
            grad.realize().expect("realize gradient failed");

            let grad_values = grad.to_vec().expect("to_vec failed");
            // Expected: 2 * x = [2, 4, 6, 8]
            assert_eq!(grad_values, vec![2.0, 4.0, 6.0, 8.0]);
        }

        #[test]
        fn test_autograd_backward_mul() {
            // Test: z = a * b, loss = sum(z)
            // Expected: dz/da = b, dz/db = a
            if !setup_metal() {
                println!("Metal not available, skipping test");
                return;
            }

            let a: Tensor<D1, f32> = Tensor::input([4]);
            let b: Tensor<D1, f32> = Tensor::input([4]);
            a.requires_grad_(true);
            b.requires_grad_(true);

            a.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data a failed");
            b.set_data(&[5.0f32, 6.0, 7.0, 8.0])
                .expect("set_data b failed");

            let z = &a * &b;
            let loss = z.sum(0);

            loss.backward_with_params(&[&a, &b])
                .expect("backward failed");

            // Get gradients
            let grad_a = a.grad().expect("gradient a should exist");
            let grad_b = b.grad().expect("gradient b should exist");

            grad_a.realize().expect("realize grad_a failed");
            grad_b.realize().expect("realize grad_b failed");

            let grad_a_values = grad_a.to_vec().expect("to_vec grad_a failed");
            let grad_b_values = grad_b.to_vec().expect("to_vec grad_b failed");

            // grad_a = b = [5, 6, 7, 8]
            // grad_b = a = [1, 2, 3, 4]
            assert_eq!(grad_a_values, vec![5.0, 6.0, 7.0, 8.0]);
            assert_eq!(grad_b_values, vec![1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_autograd_zero_grad() {
            if !setup_metal() {
                println!("Metal not available, skipping test");
                return;
            }

            let x: Tensor<D1, f32> = Tensor::input([4]);
            x.requires_grad_(true);
            x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
                .expect("set_data failed");

            // First backward
            let y1 = &x * &x;
            let loss1 = y1.sum(0);
            loss1
                .backward_with_params(&[&x])
                .expect("backward 1 failed");

            let grad1 = x.grad().expect("grad1 should exist");
            grad1.realize().expect("realize grad1 failed");
            let grad1_values = grad1.to_vec().expect("to_vec failed");
            assert_eq!(grad1_values, vec![2.0, 4.0, 6.0, 8.0]);

            // Zero grad and compute again
            x.zero_grad();
            assert!(x.grad().is_none());

            let y2 = &x + &x; // y = 2x, dy/dx = 2
            let loss2 = y2.sum(0);
            loss2
                .backward_with_params(&[&x])
                .expect("backward 2 failed");

            let grad2 = x.grad().expect("grad2 should exist");
            grad2.realize().expect("realize grad2 failed");
            let grad2_values = grad2.to_vec().expect("to_vec failed");
            // grad = 2 (ones broadcast)
            assert_eq!(grad2_values, vec![2.0, 2.0, 2.0, 2.0]);
        }

        #[test]
        fn test_kernel_cache_hit() {
            use eclat::backend::get_cache_stats;

            if !setup_metal() {
                println!("Metal not available, skipping test");
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
