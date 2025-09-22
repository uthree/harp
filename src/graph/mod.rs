use std::collections::BTreeMap;
use std::rc::{Rc, Weak};

use crate::ast::{AstNode, ConstLiteral, DType};
pub mod shape;
use crate::graph::shape::{view::View, Expr as ShapeExpr};

#[derive(Debug)]
pub struct Graph {
    nodes: BTreeMap<usize, GraphNode>,
    next_node_id: usize,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            next_node_id: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_node(&mut self, op: GraphOp) -> NodeId {
        let id = NodeId(self.next_node_id);
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
        self.nodes.get(&id.0)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id.0)
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
            GraphOp::Elementwise(_, inputs) => inputs.clone(),
            GraphOp::Cast(_, input) => vec![*input],
        })
    }

    pub fn remove_node(&mut self, id: NodeId) -> Option<GraphNode> {
        self.nodes.remove(&id.0)
    }

    pub fn cast(&mut self, input: NodeId, dtype: DType) -> NodeId {
        self.add_node(GraphOp::Cast(dtype, input))
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Add, vec![lhs, rhs]);
        self.add_node(op)
    }

    pub fn neg(&mut self, input: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Neg, vec![input]);
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
pub struct NodeId(usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub enum GraphOp {
    Input,
    Elementwise(ElementwiseOp, Vec<NodeId>), // apply element-wise operation
    Cast(DType, NodeId),                     // convert type
}

#[derive(Debug)]
pub enum ElementwiseOp {
    Add,
    Neg,
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
}
