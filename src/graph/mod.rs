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

#[derive(Debug, Clone)]
pub struct GraphNodeData {
    dtype: DType,
    op: GraphOp,
    src: Vec<GraphNode>, // 入力ノード
    view: View,
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

// AstNoderのDTypeとは異なり、VecやPtrは扱わない。
#[derive(Debug, Clone)]
pub enum DType {
    Unknown, // 未定または未知, プレースホルダー
    F32,
}

impl Graph {
    // 初期化
    fn new() -> Self {
        todo!()
    }

    // 入力ノードを新規作成, builderパターンを使う
    fn input(&mut self, name: &str) -> InputNodeBuilder {
        todo!()
    }

    // 出力ノードを登録
    fn output(&mut self, name: &str, output_node: GraphNode) {
        todo!()
    }
}

// TODO: ビルダーパターンの実装
pub struct InputNodeBuilder {}

impl InputNodeBuilder {
    pub fn with_dtype(&mut self, dtype: DType) -> Self {
        todo!()
    }

    // TIPS: 入力ノードの形状(View)は必ずContinguousである必要がある。
    pub fn with_shape(&mut self, dtype: DType) -> Self {
        todo!()
    }
}

// .0 のように書かなくても内部のデータを読み取れるようにする
impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
