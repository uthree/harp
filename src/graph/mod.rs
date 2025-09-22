use std::collections::BTreeMap;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ops::{Add, Sub, Mul, Div, Rem, Neg};

use crate::ast::{AstNode, ConstLiteral, DType};
pub mod shape;
use crate::graph::shape::{view::View, Expr as ShapeExpr};

static NEXT_GRAPH_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(usize);

impl GraphId {
    fn new() -> Self {
        GraphId(NEXT_GRAPH_ID.fetch_add(1, Ordering::SeqCst))
    }
}

#[derive(Debug)]
pub struct Graph {
    id: GraphId,
    nodes: BTreeMap<usize, GraphNode>,
    next_node_id: usize,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            id: GraphId::new(),
            nodes: BTreeMap::new(),
            next_node_id: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }


    pub fn add_node(&mut self, op: GraphOp) -> NodeId {
        let id = NodeId::new(self.id, self.next_node_id);
        self.nodes.insert(self.next_node_id, GraphNode { op });
        self.next_node_id += 1;
        id
    }

    pub fn input(&mut self) -> NodeId {
        let id = self.add_node(GraphOp::Input);
        self.inputs.push(id);
        id
    }

    pub fn output(&mut self, node_id: NodeId) {
        self.outputs.push(node_id);
    }

    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        if id.graph_id != self.id {
            panic!("Node {:?} does not belong to this graph", id);
        }
        self.nodes.get(&id.index)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        if id.graph_id != self.id {
            panic!("Node {:?} does not belong to this graph", id);
        }
        self.nodes.get_mut(&id.index)
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn node_inputs(&self, id: NodeId) -> Option<Vec<NodeId>> {
        self.get_node(id).map(|node| match &node.op {
            GraphOp::Input => vec![],
            GraphOp::Elementwise(elementwise_op) => match elementwise_op {
                // Binary operations
                ElementwiseOp::Add(lhs, rhs) => vec![*lhs, *rhs],
                ElementwiseOp::Sub(lhs, rhs) => vec![*lhs, *rhs],
                ElementwiseOp::Mul(lhs, rhs) => vec![*lhs, *rhs],
                ElementwiseOp::Div(lhs, rhs) => vec![*lhs, *rhs],
                ElementwiseOp::Max(lhs, rhs) => vec![*lhs, *rhs],
                ElementwiseOp::Rem(lhs, rhs) => vec![*lhs, *rhs],

                // Unary operations
                ElementwiseOp::Neg(input) => vec![*input],
                ElementwiseOp::Recip(input) => vec![*input],
                ElementwiseOp::Sin(input) => vec![*input],
                ElementwiseOp::Sqrt(input) => vec![*input],
                ElementwiseOp::Log2(input) => vec![*input],
                ElementwiseOp::Exp2(input) => vec![*input],
            },
            GraphOp::Cast(_, input) => vec![*input],
        })
    }

    fn validate_node(&self, id: NodeId) {
        if id.graph_id != self.id {
            panic!("Node {:?} does not belong to this graph", id);
        }
    }

    fn validate_nodes(&self, nodes: &[NodeId]) {
        for &node in nodes {
            self.validate_node(node);
        }
    }

    pub fn remove_node(&mut self, id: NodeId) -> Option<GraphNode> {
        self.validate_node(id);
        self.nodes.remove(&id.index)
    }

    pub fn cast(&mut self, input: NodeId, dtype: DType) -> NodeId {
        self.validate_node(input);
        self.add_node(GraphOp::Cast(dtype, input))
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.validate_nodes(&[lhs, rhs]);
        let op = GraphOp::Elementwise(ElementwiseOp::Add(lhs, rhs));
        self.add_node(op)
    }

    pub fn neg(&mut self, input: NodeId) -> NodeId {
        self.validate_node(input);
        let op = GraphOp::Elementwise(ElementwiseOp::Neg(input));
        self.add_node(op)
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.validate_nodes(&[lhs, rhs]);
        let op = GraphOp::Elementwise(ElementwiseOp::Sub(lhs, rhs));
        self.add_node(op)
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.validate_nodes(&[lhs, rhs]);
        let op = GraphOp::Elementwise(ElementwiseOp::Mul(lhs, rhs));
        self.add_node(op)
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.validate_nodes(&[lhs, rhs]);
        let op = GraphOp::Elementwise(ElementwiseOp::Div(lhs, rhs));
        self.add_node(op)
    }

    pub fn max(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.validate_nodes(&[lhs, rhs]);
        let op = GraphOp::Elementwise(ElementwiseOp::Max(lhs, rhs));
        self.add_node(op)
    }

    pub fn rem(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.validate_nodes(&[lhs, rhs]);
        let op = GraphOp::Elementwise(ElementwiseOp::Rem(lhs, rhs));
        self.add_node(op)
    }

    pub fn recip(&mut self, input: NodeId) -> NodeId {
        self.validate_node(input);
        let op = GraphOp::Elementwise(ElementwiseOp::Recip(input));
        self.add_node(op)
    }

    pub fn sin(&mut self, input: NodeId) -> NodeId {
        self.validate_node(input);
        let op = GraphOp::Elementwise(ElementwiseOp::Sin(input));
        self.add_node(op)
    }

    pub fn sqrt(&mut self, input: NodeId) -> NodeId {
        self.validate_node(input);
        let op = GraphOp::Elementwise(ElementwiseOp::Sqrt(input));
        self.add_node(op)
    }

    pub fn log2(&mut self, input: NodeId) -> NodeId {
        self.validate_node(input);
        let op = GraphOp::Elementwise(ElementwiseOp::Log2(input));
        self.add_node(op)
    }

    pub fn exp2(&mut self, input: NodeId) -> NodeId {
        self.validate_node(input);
        let op = GraphOp::Elementwise(ElementwiseOp::Exp2(input));
        self.add_node(op)
    }

    fn signature(&self) -> GraphSignature {
        todo!()
    }
}

#[derive(Debug)]
pub struct GraphNode {
    op: GraphOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    graph_id: GraphId,
    index: usize,
}

impl NodeId {
    fn new(graph_id: GraphId, index: usize) -> Self {
        Self { graph_id, index }
    }

    pub fn index(self) -> usize {
        self.index
    }

    pub fn graph_id(self) -> GraphId {
        self.graph_id
    }
}

/// A builder pattern for creating computation graphs with ergonomic syntax
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }

    pub fn input(&mut self) -> NodeRef {
        let id = self.graph.input();
        NodeRef { id }
    }

    pub fn build(self) -> Graph {
        self.graph
    }

    pub fn add(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let id = self.graph.add(lhs.id, rhs.id);
        NodeRef { id }
    }

    pub fn sub(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let id = self.graph.sub(lhs.id, rhs.id);
        NodeRef { id }
    }

    pub fn mul(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let id = self.graph.mul(lhs.id, rhs.id);
        NodeRef { id }
    }

    pub fn div(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let id = self.graph.div(lhs.id, rhs.id);
        NodeRef { id }
    }

    pub fn rem(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let id = self.graph.rem(lhs.id, rhs.id);
        NodeRef { id }
    }

    pub fn max(&mut self, lhs: NodeRef, rhs: NodeRef) -> NodeRef {
        let id = self.graph.max(lhs.id, rhs.id);
        NodeRef { id }
    }

    pub fn neg(&mut self, input: NodeRef) -> NodeRef {
        let id = self.graph.neg(input.id);
        NodeRef { id }
    }

    pub fn recip(&mut self, input: NodeRef) -> NodeRef {
        let id = self.graph.recip(input.id);
        NodeRef { id }
    }

    pub fn sin(&mut self, input: NodeRef) -> NodeRef {
        let id = self.graph.sin(input.id);
        NodeRef { id }
    }

    pub fn sqrt(&mut self, input: NodeRef) -> NodeRef {
        let id = self.graph.sqrt(input.id);
        NodeRef { id }
    }

    pub fn log2(&mut self, input: NodeRef) -> NodeRef {
        let id = self.graph.log2(input.id);
        NodeRef { id }
    }

    pub fn exp2(&mut self, input: NodeRef) -> NodeRef {
        let id = self.graph.exp2(input.id);
        NodeRef { id }
    }
}

/// A reference to a node that can be used for ergonomic operations
#[derive(Clone, Copy)]
pub struct NodeRef {
    id: NodeId,
}

impl NodeRef {
    pub fn id(self) -> NodeId {
        self.id
    }

}


#[derive(Debug)]
pub enum GraphOp {
    Input,
    Elementwise(ElementwiseOp), // apply element-wise operation
    Cast(DType, NodeId),        // convert type
}

#[derive(Debug)]
pub enum ElementwiseOp {
    // Binary operations
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Max(NodeId, NodeId),
    Rem(NodeId, NodeId),

    // Unary operations
    Neg(NodeId),
    Recip(NodeId),
    Sin(NodeId),
    Sqrt(NodeId),
    Log2(NodeId),
    Exp2(NodeId),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let mut graph = Graph::new();

        // 入力ノードを2つ作成
        let input1 = graph.input();
        let input2 = graph.input();

        // 加算ノードを作成
        let add_node = graph.add(input1, input2);

        // 否定ノードを作成
        let neg_node = graph.neg(add_node);

        // 出力として登録
        graph.output(neg_node);

        // 構造の確認
        assert_eq!(graph.inputs().len(), 2);
        assert_eq!(graph.outputs().len(), 1);
        assert_eq!(graph.nodes.len(), 4);

        // add_nodeの入力確認（削除前）
        let add_inputs = graph.node_inputs(add_node).unwrap();
        assert_eq!(add_inputs, vec![input1, input2]);

        // neg_nodeの入力確認（削除前）
        let neg_inputs = graph.node_inputs(neg_node).unwrap();
        assert_eq!(neg_inputs, vec![add_node]);

        // ノード削除のテスト
        let removed_node = graph.remove_node(add_node);
        assert!(removed_node.is_some());
        assert_eq!(graph.nodes.len(), 3);

        // 削除されたノードは取得できない
        assert!(graph.get_node(add_node).is_none());
    }

    #[test]
    fn test_arithmetic_operations() {
        let mut graph = Graph::new();

        // 入力ノードを作成
        let input1 = graph.input();
        let input2 = graph.input();

        // 各種算術演算をテスト
        let sub_node = graph.sub(input1, input2);
        let mul_node = graph.mul(input1, input2);
        let div_node = graph.div(input1, input2);
        let max_node = graph.max(input1, input2);
        let rem_node = graph.rem(input1, input2);

        // 単項演算をテスト
        let recip_node = graph.recip(input1);
        let sin_node = graph.sin(input1);
        let sqrt_node = graph.sqrt(input1);
        let log2_node = graph.log2(input1);
        let exp2_node = graph.exp2(input1);

        // 入力数の確認
        assert_eq!(graph.node_inputs(sub_node).unwrap(), vec![input1, input2]);
        assert_eq!(graph.node_inputs(mul_node).unwrap(), vec![input1, input2]);
        assert_eq!(graph.node_inputs(div_node).unwrap(), vec![input1, input2]);
        assert_eq!(graph.node_inputs(max_node).unwrap(), vec![input1, input2]);
        assert_eq!(graph.node_inputs(rem_node).unwrap(), vec![input1, input2]);

        assert_eq!(graph.node_inputs(recip_node).unwrap(), vec![input1]);
        assert_eq!(graph.node_inputs(sin_node).unwrap(), vec![input1]);
        assert_eq!(graph.node_inputs(sqrt_node).unwrap(), vec![input1]);
        assert_eq!(graph.node_inputs(log2_node).unwrap(), vec![input1]);
        assert_eq!(graph.node_inputs(exp2_node).unwrap(), vec![input1]);

        // ノード数の確認
        assert_eq!(graph.nodes.len(), 12); // 2入力 + 10演算
    }

    #[test]
    #[should_panic(expected = "Node")]
    fn test_cross_graph_node_access() {
        let mut graph1 = Graph::new();
        let mut graph2 = Graph::new();

        let node1 = graph1.input();
        let _node2 = graph2.input();

        // 異なるグラフのノードにアクセスしようとするとパニックする
        graph2.get_node(node1);
    }

    #[test]
    #[should_panic(expected = "Node")]
    fn test_cross_graph_operation() {
        let mut graph1 = Graph::new();
        let mut graph2 = Graph::new();

        let node1 = graph1.input();
        let node2 = graph2.input();

        // 異なるグラフのノード同士で演算しようとするとパニックする
        graph1.add(node1, node2);
    }

    #[test]
    #[should_panic(expected = "Node")]
    fn test_cross_graph_unary_operation() {
        let mut graph1 = Graph::new();
        let mut graph2 = Graph::new();

        let node1 = graph1.input();
        let _node2 = graph2.input();

        // 異なるグラフのノードで単項演算しようとするとパニックする
        graph2.neg(node1);
    }

    #[test]
    fn test_same_graph_operations() {
        let mut graph = Graph::new();

        let node1 = graph.input();
        let node2 = graph.input();

        // 同じグラフ内のノード同士の演算は正常に動作する
        let add_result = graph.add(node1, node2);
        let neg_result = graph.neg(node1);

        assert_eq!(graph.node_inputs(add_result).unwrap(), vec![node1, node2]);
        assert_eq!(graph.node_inputs(neg_result).unwrap(), vec![node1]);
    }

    #[test]
    fn test_graph_builder() {
        let mut builder = GraphBuilder::new();

        // 入力ノードを作成
        let a = builder.input();
        let b = builder.input();

        // 各種演算を実行
        let sum = builder.add(a, b);
        let product = builder.mul(sum, a);
        let sin_result = builder.sin(product);
        let final_result = builder.sqrt(sin_result);

        // グラフを構築
        let graph = builder.build();

        // グラフが正しく構築されていることを確認
        assert_eq!(graph.nodes.len(), 6); // 2入力 + 4演算
        assert_eq!(graph.inputs().len(), 2);

        // 最終ノードの入力を確認
        let inputs = graph.node_inputs(final_result.id()).unwrap();
        assert_eq!(inputs.len(), 1);
    }

    #[test]
    fn test_builder_unary_operations() {
        let mut builder = GraphBuilder::new();

        let x = builder.input();

        // 単項演算のテスト
        let neg_x = builder.neg(x);
        let recip_x = builder.recip(x);
        let sin_x = builder.sin(x);
        let sqrt_x = builder.sqrt(x);
        let log2_x = builder.log2(x);
        let exp2_x = builder.exp2(x);

        let graph = builder.build();

        // 各ノードが正しく作成されていることを確認
        assert_eq!(graph.nodes.len(), 7); // 1入力 + 6演算
        assert!(graph.get_node(neg_x.id()).is_some());
        assert!(graph.get_node(recip_x.id()).is_some());
        assert!(graph.get_node(sin_x.id()).is_some());
        assert!(graph.get_node(sqrt_x.id()).is_some());
        assert!(graph.get_node(log2_x.id()).is_some());
        assert!(graph.get_node(exp2_x.id()).is_some());
    }
}
