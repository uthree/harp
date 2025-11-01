use crate::{ast::Literal, graph::shape::View};
use std::{
    collections::HashMap,
    rc::{Rc, Weak},
};
pub mod shape;

#[derive(Debug)]
pub struct Graph {
    inputs: HashMap<String, Weak<GraphNodeData>>, // Rcの参照カウントに影響を与えないために、Weak参照で保持する。
    outputs: HashMap<String, GraphNode>,
}

#[derive(Debug)]
pub enum GraphOp {
    Input,                      // 入力ノード
    Contiguous,                 // Viewに従って要素を並べ直す。
    Const(Literal),             // 定数ノード, shape=[], ndim=0のスカラーを初期化する。
    View(View),                 // Viewを変更する
    Elementwise(ElementwiseOp), // 要素ごとに演算を行う
    Reduce,                     // 縮約
    Cumulative,                 // 累積
}

#[derive(Debug)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Max,
    Rem,
    Idiv,
    Neg,
    Recip,
}

#[derive(Debug)]
pub struct GraphNode(Rc<GraphNodeData>);

#[derive(Debug)]
pub struct GraphNodeData {
    dtype: DType,
    op: GraphOp,
}

// AstNoderのDTypeとは異なり、VecやPtrは扱わない。
#[derive(Debug)]
pub enum DType {
    F32,
}
