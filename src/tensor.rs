use crate::dtype::DType;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
enum Ops {}

#[derive(Debug, Clone, PartialEq)]
struct Tensor_ {
    op: Ops,
    dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor(Rc<Tensor_>);
