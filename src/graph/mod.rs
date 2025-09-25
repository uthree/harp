use std::collections::BTreeMap;
use std::rc::{Rc, Weak};

use crate::ast::{AstNode, ConstLiteral, DType};
pub mod shape;
use crate::graph::shape::{view::View, Expr as ShapeExpr};

#[derive(Debug)]
pub struct GraphNodeData {}

#[derive(Debug)]
pub struct GraphNode(Rc<GraphNodeData>);

#[derive(Debug)]
pub struct Graph {}

#[derive(Debug)]
pub enum GraphOp {
    Input,
    Const(ConstLiteral), // initialize single element tensor, shape=[],
}

#[derive(Debug)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>,
    pub inputs: Vec<TensorSignature>,
    pub outputs: Vec<TensorSignature>,
}

#[derive(Debug)]
pub struct ShapeVariableSignature {
    pub name: String,
    pub default: isize,
}

#[derive(Debug)]
pub struct TensorSignature {
    pub dtype: DType,
    pub shape: Vec<ShapeExpr>,
}
