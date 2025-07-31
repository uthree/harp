use crate::uop::{Const, DType, Op, UOp};
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
    Elementwise(Op),
    Reduce(Op, usize),
    Cumulative(Op, usize),
    Contiguous,
}
