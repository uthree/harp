use crate::{node::Node, operator, shape::{symbolic::Expr, tracker::ShapeTracker}, tensor::Tensor};
use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct Graph {
    pub graph: DiGraph<Node, usize>,
    pub outputs: Vec<NodeIndex>,
    pub inputs: Vec<NodeIndex>,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            graph: DiGraph::new(),
            outputs: Vec::new(),
            inputs: Vec::new(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_input(graph: Arc<Mutex<Self>>, shape: ShapeTracker) -> Tensor {
        let mut graph_mut = graph.lock().unwrap();
        let node = Node::new(operator::Input, shape.clone());
        let node_index = graph_mut.add_node(node);
        graph_mut.inputs.push(node_index);

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

    pub fn add_output(&mut self, tensor: &Tensor) {
        self.outputs.push(tensor.node_index);
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

    pub fn parents(&self, index: NodeIndex) -> impl Iterator<Item = (NodeIndex, usize)> + '_ {
        self.graph
            .edges_directed(index, Direction::Incoming)
            .map(|edge| (edge.source(), *edge.weight()))
    }

    pub fn to_dot(&self) -> String {
        let graph = &self.graph;
        let outputs = &self.outputs;
        let inputs = &self.inputs;

        petgraph::dot::Dot::with_attr_getters(
            graph,
            &[], // No global config
            &|_, edge_data| {
                format!("label = \"{}\"", edge_data.weight())
            },
            &|_, node_data: (NodeIndex, &Node)| {
                let id = node_data.0;
                let node = node_data.1;

                // Create a temporary ShapeTracker with replaced Index expressions
                let mut temp_shape = node.shape.clone();
                for (i, expr) in temp_shape.map.iter_mut().enumerate() {
                    *expr = expr.clone().replace(&Expr::Index, &Expr::Var(format!("idx{}", i)));
                }
                for (i, expr) in temp_shape.max.iter_mut().enumerate() {
                    *expr = expr.clone().replace(&Expr::Index, &Expr::Var(format!("idx{}", i)));
                }

                let mut attrs = vec![format!("label = \"{:?}\n{}\"", node.op(), temp_shape)];
                if outputs.contains(&id) {
                    attrs.push("peripheries=2".to_string());
                }
                else if inputs.contains(&id) {
                    attrs.push("style=filled".to_string());
                    attrs.push("fillcolor=lightgray".to_string());
                }
                attrs.join(", ")
            },
        ).to_string()
    }

    pub fn optimize(&mut self) {
        // Implement optimization passes here, such as:
        // - Dead code elimination
        // - Constant folding
        // - Operator fusion
    }
}
