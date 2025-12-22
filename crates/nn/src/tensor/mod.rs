pub mod initialization;

pub use initialization::{TensorInit, TensorRandInit};

use harp_autograd::Differentiable;
use harp_lazy_array::LazyArray;
use harp_lazy_array::prelude::*;

/// Tensor型: 自動微分可能な遅延評価配列
pub type Tensor<T, D> = Differentiable<LazyArray<T, D>>;
pub type Tensor0<T> = Tensor<T, Dim0>;
pub type Tensor1<T> = Tensor<T, Dim1>;
pub type Tensor2<T> = Tensor<T, Dim2>;
pub type Tensor3<T> = Tensor<T, Dim3>;
pub type Tensor4<T> = Tensor<T, Dim4>;
pub type Tensor5<T> = Tensor<T, Dim5>;
pub type Tensor6<T> = Tensor<T, Dim6>;
pub type TensorD<T> = Tensor<T, DimDyn>;
