//! Tensor初期化用の拡張トレイト

use harp_autograd::Differentiable;
use harp_lazy_array::{Dimension, IntoShape, LazyArray};

/// Tensor初期化用の拡張トレイト
pub trait TensorInit: Sized {
    fn zeros(shape: impl IntoShape) -> Self;
    fn ones(shape: impl IntoShape) -> Self;
}

/// 乱数初期化用の拡張トレイト（f32専用）
pub trait TensorRandInit: Sized {
    /// 一様乱数 [0, 1) で初期化されたテンソルを作成
    fn rand(shape: impl IntoShape) -> Self;
}

macro_rules! impl_tensor_init {
    ($target:ty) => {
        impl<D: Dimension> TensorInit for Differentiable<LazyArray<$target, D>> {
            fn zeros(shape: impl IntoShape) -> Self {
                Differentiable::new(LazyArray::<$target, D>::zeros(shape))
            }

            fn ones(shape: impl IntoShape) -> Self {
                Differentiable::new(LazyArray::<$target, D>::ones(shape))
            }
        }
    };
}

impl_tensor_init!(i32);
impl_tensor_init!(f32);

// TensorRandInit は f32 専用
impl<D: Dimension> TensorRandInit for Differentiable<LazyArray<f32, D>> {
    fn rand(shape: impl IntoShape) -> Self {
        Differentiable::new(LazyArray::<f32, D>::rand(shape))
    }
}
