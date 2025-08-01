use crate::ast::{AstNode, DType, Op as AstOp};
use std::cell::Cell;
use std::ops::Deref;
use std::rc::Rc;

thread_local! {
    static NEXT_ID: Cell<usize> = Cell::new(0);
}

fn next_id() -> usize {
    NEXT_ID.with(|cell| {
        let id = cell.get();
        cell.set(id + 1);
        id
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    op: TensorOp,
    src: Vec<Tensor>,
    dtype: DType,
    id: usize,
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<TensorData>);

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.src == other.src
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    Elementwise(AstNode),
    Reduce(AstOp, Vec<usize>),
    Contiguous,
}

impl Deref for Tensor {
    type Target = TensorData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn new(op: TensorOp, src: Vec<Tensor>, dtype: DType) -> Tensor {
        Tensor(Rc::new(TensorData {
            op: op,
            src: src,
            dtype: dtype,
            id: next_id(),
        }))
    }
}
