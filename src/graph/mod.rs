use crate::ast::{ConstLiteral, DType};
pub mod ops;
pub mod shape;
use crate::graph::shape::Expr as ShapeExpr;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

#[derive(Debug, Clone)]
pub struct GraphNodeData {
    pub op: GraphOp,
    pub src: Vec<GraphNode>,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub inputs: Vec<GraphNode>,
    pub outputs: Vec<GraphNode>,
    pub shape_variables: Vec<ShapeVariableSignature>,
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

#[derive(Debug, Clone)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Max,
    Neg,
    Recip,
    Sqrt,
    Sin,
    Log2,
    Exp2,
    Rem,
}

#[derive(Debug, Clone)]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

#[derive(Debug, Clone)]
pub enum GraphOp {
    Input,
    Const(ConstLiteral, Vec<ShapeExpr>),
    Elementwise(ElementwiseOp),
    Cast,
    Rand(Vec<ShapeExpr>),
    Arange(usize),
    Reshape(Vec<ShapeExpr>),
    Reduce(ReduceOp, usize),
    Cumulative(ReduceOp, usize),
    Contiguous,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            inputs: vec![],
            outputs: vec![],
            shape_variables: vec![],
        }
    }

    // 自身のシグネチャを返す
    pub fn signature() -> GraphSignature {
        todo!()
    }
}

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
