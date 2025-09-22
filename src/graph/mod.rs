use std::rc::{Rc, Weak};

use crate::ast::{AstNode, ConstLiteral, DType};
pub mod shape;
use crate::graph::shape::{view::View, Expr as ShapeExpr};

#[derive(Debug)]
pub struct Graph {
    nodes: Vec<GraphNode>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_node(&mut self, op: GraphOp, inputs: Vec<NodeId>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(GraphNode { op, inputs });
        id
    }

    pub fn add_input(&mut self) -> NodeId {
        let id = self.add_node(GraphOp::Input, vec![]);
        self.inputs.push(id);
        id
    }

    pub fn add_output(&mut self, node_id: NodeId) {
        self.outputs.push(node_id);
    }

    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id.0)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id.0)
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn node_inputs(&self, id: NodeId) -> Option<&[NodeId]> {
        self.get_node(id).map(|node| &node.inputs[..])
    }

    pub fn add_cast(&mut self, input: NodeId, dtype: DType) -> NodeId {
        self.add_node(GraphOp::Cast(dtype), vec![input])
    }

    pub fn add_add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Add(lhs, rhs));
        self.add_node(op, vec![lhs, rhs])
    }

    pub fn add_neg(&mut self, input: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Neg(input));
        self.add_node(op, vec![input])
    }

    fn signature(&self) -> GraphSignature {
        todo!()
    }
}

#[derive(Debug)]
pub struct GraphNode {
    op: GraphOp,
    inputs: Vec<NodeId>,
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
    Elementwise(ElementwiseOp), // apply element-wise operation
    Cast(DType),                // convert type
}

#[derive(Debug)]
pub enum ElementwiseOp {
    Add(NodeId, NodeId),
    Neg(NodeId),
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
        let input1 = graph.add_input();
        let input2 = graph.add_input();

        // 加算ノードを作成
        let add_node = graph.add_add(input1, input2);

        // 否定ノードを作成
        let neg_node = graph.add_neg(add_node);

        // 出力として登録
        graph.add_output(neg_node);

        // 構造の確認
        assert_eq!(graph.inputs().len(), 2);
        assert_eq!(graph.outputs().len(), 1);
        assert_eq!(graph.nodes.len(), 4);

        // add_nodeの入力確認
        let add_inputs = graph.node_inputs(add_node).unwrap();
        assert_eq!(add_inputs, &[input1, input2]);

        // neg_nodeの入力確認
        let neg_inputs = graph.node_inputs(neg_node).unwrap();
        assert_eq!(neg_inputs, &[add_node]);
    }
}
