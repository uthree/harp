use std::rc::Rc;

use crate::ast::{AstNode, ConstLiteral, DType};
pub mod shape;
use crate::graph::shape::{view::View, Expr as ShapeExpr};

#[derive(Debug)]
pub struct Graph {}

impl Graph {
    fn signature(&self) -> GraphSignature {
        todo!()
    }
}

pub struct GraphNodeData {
    op: GraphOp,
}
pub struct GraphNode(Rc<GraphNodeData>);

pub enum GraphOp {
    Input { dtype: DType, shape: Vec<ShapeExpr> },
}

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
