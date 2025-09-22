use std::rc::{Rc, Weak};

use crate::ast::{AstNode, ConstLiteral, DType};
pub mod shape;
use crate::graph::shape::{view::View, Expr as ShapeExpr};

#[derive(Debug)]
pub struct Graph {
    nodes: Vec<GraphNode>,
}

impl Graph {
    fn signature(&self) -> GraphSignature {
        todo!()
    }
}

#[derive(Debug)]
pub struct GraphNode {
    op: GraphOp,
}

#[derive(Debug)]
pub struct NodeId(usize);

#[derive(Debug)]
pub enum GraphOp {
    Input,
    Elementwise(ElementwiseOp), // apply element-wise operation
    Cast(DType),                // convert type
}

#[derive(Debug)]
pub enum ElementwiseOp {}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>,
    pub inputs: Vec<TensorSignature>,
    pub outputs: Vec<TensorSignature>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    pub name: String,
    pub default: isize,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct TensorSignature {
    pub dtype: DType,
    pub shape: Vec<ShapeExpr>,
}
