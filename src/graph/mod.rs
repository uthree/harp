use crate::{
    ast::Literal,
    graph::{ops::GraphOp, shape::View},
};
use std::{
    collections::HashMap,
    ops::Deref,
    rc::{Rc, Weak},
};
pub mod ops;
pub mod shape;

#[derive(Debug)]
pub struct Graph {
    inputs: HashMap<String, Weak<GraphNodeData>>, // Rcの参照カウントに影響を与えないために、Weak参照で保持する。
    outputs: HashMap<String, GraphNode>,
}

#[derive(Debug)]
pub struct GraphNode(Rc<GraphNodeData>);

#[derive(Debug)]
pub struct GraphNodeData {
    dtype: DType,
    op: GraphOp,
    src: Vec<GraphNode>, // 入力ノード
}

// AstNoderのDTypeとは異なり、VecやPtrは扱わない。
#[derive(Debug)]
pub enum DType {
    Unknown, // 未定または未知, プレースホルダー
    F32,
}

impl Graph {
    // 初期化
    fn new() -> Self {
        todo!()
    }

    // 入力ノードを新規作成
    fn input(&mut self, name: &str) -> GraphNode {
        todo!()
    }

    // 出力ノードを登録
    fn output(&mut self, name: &str, output_node: GraphNode) {
        todo!()
    }
}

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl GraphNode {
    // 入力ノードに型を指定
    fn with_dtype(&mut self, dtype: DType) {
        todo!()
    }
}
