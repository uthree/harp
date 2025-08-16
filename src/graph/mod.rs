pub mod shape;

use crate::ast::{AstOp, DType};
use crate::graph::shape::expr::Expr as ShapeExpr;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone, PartialEq)]
pub struct GraphSignature {
    shape_variables: Vec<ShapeVariableSignature>, // Shapeを決定するための変数。
    inputs: Vec<TensorSignature>,                 // 入力の型
    outputs: Vec<TensorSignature>,                // 出力の型
}

// Shapeを決定するのに使う変数（整数）のシグネチャ。これを導入することにより、異なるサイズのテンソルであっても、同じカーネルや計算グラフを流用できる。
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    name: String,         // 変数名
    condition: ShapeExpr, // その値が利用可能かどうか判定するための式
    default: isize,       // デフォルト値, ベンチマークや最適化のために使用する。
}

// 入出力テンソルの型を表現する構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSignature {
    dtype: DType, // データ型
    shape: Vec<ShapeExpr>, // 形状
                  // ちなみにViewに関しては、入出力の時点では常にContiguousであるとする。
}

#[derive(Debug, Clone, PartialEq)]
pub enum GraphOp {
    Input { shape: Vec<ShapeExpr>, dtype: DType },
    Contiguous,
    Elementwise(AstOp),   // apply element-wise operator
    Reduce(AstOp, usize), // reduce dimension
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNodeData {
    op: GraphOp,
    src: Vec<GraphNode>,
    dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode(Rc<GraphNodeData>);

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    inputs: Vec<GraphNode>,
}
