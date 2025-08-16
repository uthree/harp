pub mod shape;

use crate::ast::{AstOp, DType};
use crate::graph::shape::expr::Expr as ShapeExpr;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub struct GraphSignature {
    shape_variables: Vec<ShapeVariableSignature>,
    inputs: Vec<BufferSignature>,  // 入力の型
    outputs: Vec<BufferSignature>, // 出力の型
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    name: String,         // 変数名
    condition: ShapeExpr, // その値が利用可能かどうか判定するための式
    default: isize,       // デフォルト値
}

// 入出力バッファーの型を表現する構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct BufferSignature {
    dtype: DType,
    shape: Vec<ShapeExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GraphOp {
    Input { shape: Vec<ShapeExpr>, dtype: DType },
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
