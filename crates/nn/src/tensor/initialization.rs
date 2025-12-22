use crate::tensor::Tensor;
use harp_autograd::Differentiable;
use harp_lazy_array::{Dimension, IntoShape, LazyArray};

macro_rules! impl_zeros_and_ones_init {
    ($target:ty) => {
        impl<D> Tensor<$target, D>
        where
            D: Dimension,
        {
            fn zeros(shape: impl IntoShape) -> Self {
                Self(Differentiable::new(LazyArray::<$target, D>::zeros(
                    shape.into_shape(),
                )))
            }
        }

        impl<D> Tensor<$target, D>
        where
            D: Dimension,
        {
            fn ones(shape: impl IntoShape) -> Self {
                Self(Differentiable::new(LazyArray::<$target, D>::ones(
                    shape.into_shape(),
                )))
            }
        }
    };
}

impl_zeros_and_ones_init!(i32);
impl_zeros_and_ones_init!(f32);
