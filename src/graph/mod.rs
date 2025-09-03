use std::rc::Rc;

use crate::ast::{AstNode, ConstLiteral, DType};
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

pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

pub enum GraphOp {
    Const(ConstLiteral, Vec<ShapeExpr>), // 定数ノード, 任意のshapeの定数
    Rand(Vec<ShapeExpr>),                // 任意のshapeの一様乱数を生成
    Arange(usize), // 任意の長さの１次元Tensorを生成、 内容は [0, 1, 2, ..., n],
    Reshape(Vec<ShapeExpr>),
    Elementwise(AstNode),         // Capture(n) が src[n]に対応する要素ごとの演算
    Reduce(ReduceOp, Vec<usize>), // 1つ以上の軸をReduce
    ElemenWiseReduce(AstNode, ReduceOp, Vec<usize>), //element-wiseな演算をしながらReduceする
    Cumulative(ReduceOp, usize),  // 累積演算
    ElementwiseCumulative(AstNode, ReduceOp, usize), // Element-wise演算をしつつ累積する
    Contiguous,                   // 連続なメモリ配置に並べ直す
}

pub struct GraphNodeData {
    op: GraphOp,
    src: Vec<GraphNode>,
    dtype: DType,
}

pub struct GraphNode(Rc<GraphNodeData>);

pub struct Graph {
    pub signature: GraphSignature,
    pub inputs: Vec<GraphNode>,
    pub outputs: Vec<GraphNode>,
}
