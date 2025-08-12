use super::{DType, Shape, Tensor, TensorData, TensorOp};
use crate::{ast::Const, backend::{Backend, c::CBuffer}};
use std::sync::Arc;

impl Tensor {
    pub fn rand(
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
        backend: Arc<dyn Backend<CBuffer>>,
    ) -> Self {
        TensorData::new(TensorOp::Rand, vec![], shape, dtype, requires_grad, backend).into()
    }

    pub fn full(
        shape: Shape,
        dtype: DType,
        value: Const,
        requires_grad: bool,
        backend: Arc<dyn Backend<CBuffer>>,
    ) -> Self {
        TensorData::new(
            TensorOp::Full(value),
            vec![],
            shape,
            dtype,
            requires_grad,
            backend,
        )
        .into()
    }

    pub fn ones(
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
        backend: Arc<dyn Backend<CBuffer>>,
    ) -> Self {
        Self::full(shape, dtype, Const::from(1.0), requires_grad, backend)
    }

    pub fn zeros(
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
        backend: Arc<dyn Backend<CBuffer>>,
    ) -> Self {
        Self::full(shape, dtype, Const::from(0.0), requires_grad, backend)
    }
}
