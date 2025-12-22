pub mod initialization;

use harp_autograd::Differentiable;
use harp_lazy_array::prelude::*;
use harp_lazy_array::{ArrayElement, LazyArray};

pub struct Tensor<T: ArrayElement, D: Dimension>(Differentiable<LazyArray<T, D>>);
pub type Tensor0<T> = Tensor<T, Dim0>;
pub type Tensor1<T> = Tensor<T, Dim1>;
pub type Tensor2<T> = Tensor<T, Dim2>;
pub type Tensor3<T> = Tensor<T, Dim3>;
pub type Tensor4<T> = Tensor<T, Dim4>;
pub type Tensor5<T> = Tensor<T, Dim5>;
pub type Tensor6<T> = Tensor<T, Dim6>;
pub type TensorD<T> = Tensor<T, DimDyn>;
