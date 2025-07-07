use crate::node::Node;
use petgraph::graph::{DiGraph, NodeIndex};

#[derive(Debug, Default)]
pub struct Graph {
    graph: DiGraph<Node, usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        self.graph.add_node(node)
    }

    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, arg_index: usize) {
        self.graph.add_edge(from, to, arg_index);
    }

    pub fn optimize(&mut self) {
        // Implement optimization passes here, such as:
        // - Dead code elimination
        // - Constant folding
        // - Operator fusion
    }
}