use std::marker::PhantomData;

pub struct Tensor<T, D: Dimension> {
    _dtype: PhantomData<T>,
    _dim: PhantomData<D>,
}
