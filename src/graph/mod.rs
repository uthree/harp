pub mod shape;

use crate::ast::{AstOp, DType};
use crate::graph::shape::expr::Expr;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone, PartialEq)]
pub enum GraphOp {
    Input { shape: Vec<Expr>, dtype: DType },
    Contiguous,
    Elementwise(AstOp),   // apply element-wise operator
    Reduce(AstOp, usize), // reduce dimension
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeData {
    op: GraphOp,
    src: Vec<Node>,
    dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node(Rc<NodeData>);

impl Deref for Node {
    type Target = NodeData;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
