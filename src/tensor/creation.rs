use super::{DType, Shape, Tensor, TensorData, TensorOp, backend};
use crate::ast::Const;

impl Tensor {
    pub fn rand(shape: Shape, dtype: DType, requires_grad: bool) -> Self {
        TensorData::new(
            TensorOp::Rand,
            vec![],
            shape,
            dtype,
            requires_grad,
            backend("c"),
        )
        .into()
    }

    pub fn full(shape: Shape, dtype: DType, value: Const, requires_grad: bool) -> Self {
        TensorData::new(
            TensorOp::Full(value),
            vec![],
            shape,
            dtype,
            requires_grad,
            backend("c"),
        )
        .into()
    }

    pub fn ones(shape: Shape, dtype: DType, requires_grad: bool) -> Self {
        Self::full(shape, dtype, Const::from(1.0), requires_grad)
    }

    pub fn zeros(shape: Shape, dtype: DType, requires_grad: bool) -> Self {
        Self::full(shape, dtype, Const::from(0.0), requires_grad)
    }
}
