use crate::node::Node;
use petgraph::graph::{DiGraph, NodeIndex};

#[derive(Debug, Default)]
pub struct Graph {
    graph: DiGraph<Node, ()>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        self.graph.add_node(node)
    }

    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from, to, ());
    }

    pub fn optimize(&mut self) {
        // Implement optimization passes here, such as:
        // - Dead code elimination
        // - Constant folding
        // - Operator fusion
    }
}