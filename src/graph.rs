use crate::{
    node::Node,
    operator,
    shape::tracker::ShapeTracker,
    tensor::Tensor,
};
use petgraph::{
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
    Direction,
};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
pub struct Graph {
    graph: DiGraph<Node, usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_input(graph: Arc<Mutex<Self>>, shape: ShapeTracker) -> Tensor {
        let mut graph_mut = graph.lock().unwrap();
        let node = Node::new(operator::Input);
        let node_index = graph_mut.add_node(node);

        Tensor {
            graph: graph.clone(),
            node_index,
            shape,
        }
    }

    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        self.graph.add_node(node)
    }

    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, arg_index: usize) {
        self.graph.add_edge(from, to, arg_index);
    }

    // --- Accessors for testing and debugging ---

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn node_weight(&self, index: NodeIndex) -> Option<&Node> {
        self.graph.node_weight(index)
    }

    pub fn parents(&self, index: NodeIndex) -> impl Iterator<Item = (NodeIndex, usize)> + '_
    {
        self.graph
            .edges_directed(index, Direction::Incoming)
            .map(|edge| (edge.source(), *edge.weight()))
    }

    pub fn optimize(&mut self) {
        // Implement optimization passes here, such as:
        // - Dead code elimination
        // - Constant folding
        // - Operator fusion
    }
}