use crate::ast::{AstNode, Const, DType, Op as AstOp};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    op: TensorOp,
    dtype: DType,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Rc<TensorData>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    Elementwise(AstOp),
    Reduce(AstOp, usize),
    Cumulative(AstOp, usize),
    Contiguous,
}
