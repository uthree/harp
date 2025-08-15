pub mod shape;

use crate::ast::DType;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone, PartialEq)]
pub enum GraphOp {
    Input,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeData {
    op: GraphOp,
    src: Vec<Node>,
    dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node(Rc<NodeData>);
