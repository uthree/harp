use std::rc::Rc;

use crate::ast::{AstNode, DType};
pub mod shape;
use crate::graph::shape::Expr as ShapeExpr;

#[derive(Debug, Clone, PartialEq)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>, // Shapeを決定するための変数。
    pub inputs: Vec<TensorSignature>,                 // 入力の型
    pub outputs: Vec<TensorSignature>,                // 出力の型
}

// Shapeを決定するのに使う変数（整数）のシグチャ。これを導入することにより、異なるサイズのテンソルであっても、同じカーネルや計算グラフを流用できる。
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    pub name: String,   // 変数名
    pub default: isize, // デフォルト値, ベンチマークや最適化のために使用する。
}

// 入出力テンソルの型を表現する構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSignature {
    pub dtype: DType, // データ型
    pub shape: Vec<ShapeExpr>,
    // ちなみにViewに関しては、入出力の時点では常にContiguousであるとする。
}

pub enum ElementwiseOp {}

pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

pub enum GraphOp {
    FusedElementwise(AstNode), // capture(n) が src[n]に対応する
    FusedReduce(Vec<usize>),   // 1つ以上の軸をReduce
}

pub struct GraphNodeData {
    op: GraphOp,
    src: Vec<GraphNode>,
}

pub struct GraphNode(Rc<GraphNodeData>);

pub struct Graph {
    pub signature: GraphSignature,
    pub inputs: Vec<GraphNode>,
    pub outputs: Vec<GraphNode>,
}
