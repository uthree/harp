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

    pub fn remove_node(&mut self, id: NodeId) -> Option<GraphNode> {
        self.nodes.remove(&id.0)
    }

    pub fn cast(&mut self, input: NodeId, dtype: DType) -> NodeId {
        self.add_node(GraphOp::Cast(dtype, input))
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Add(lhs, rhs));
        self.add_node(op)
    }

    pub fn neg(&mut self, input: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Neg(input));
        self.add_node(op)
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Sub(lhs, rhs));
        self.add_node(op)
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Mul(lhs, rhs));
        self.add_node(op)
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Div(lhs, rhs));
        self.add_node(op)
    }

    pub fn max(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Max(lhs, rhs));
        self.add_node(op)
    }

    pub fn rem(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Rem(lhs, rhs));
        self.add_node(op)
    }

    pub fn recip(&mut self, input: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Recip(input));
        self.add_node(op)
    }

    pub fn sin(&mut self, input: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Sin(input));
        self.add_node(op)
    }

    pub fn sqrt(&mut self, input: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Sqrt(input));
        self.add_node(op)
    }

    pub fn log2(&mut self, input: NodeId) -> NodeId {
        let op = GraphOp::Elementwise(ElementwiseOp::Log2(input));
        self.add_node(op)
    }

    pub fn exp2(&mut self, input: NodeId) -> NodeId {
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
}
