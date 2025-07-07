use crate::{
    node::Node,
    operator,
    shape::{symbolic::Expr, tracker::ShapeTracker},
    tensor::Tensor,
};
use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
};
use std::sync::{Arc, Mutex};

/// Represents a computation graph.
///
/// A `Graph` consists of nodes (operations or data) and edges (data flow)
/// managed by a `petgraph::DiGraph`. It tracks input and output nodes
/// to define the boundaries of the computation.
#[derive(Debug)]
pub struct Graph {
    /// The underlying directed graph storing nodes and their connections.
    pub graph: DiGraph<Node, (usize, usize)>, // The 0th element of Edge indicates which argument the edge is input as, and the 1st element indicates which output it is taking out.
    /// Indices of the output nodes in the graph.
    pub outputs: Vec<NodeIndex>,
    /// Indices of the input nodes in the graph.
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
    /// Creates a new, empty computation graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::graph::Graph;
    /// let graph = Graph::new();
    /// assert_eq!(graph.node_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new input tensor to the graph.
    ///
    /// This creates a new `Node` of type `operator::Input` with the given shape,
    /// adds it to the graph, and registers it as an input node.
    ///
    /// # Arguments
    ///
    /// * `graph` - An `Arc<Mutex<Self>>` reference to the graph.
    /// * `shape` - The `ShapeTracker` defining the shape of the input tensor.
    ///
    /// # Returns
    ///
    /// A `Tensor` representing the newly created input.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use harp::graph::Graph;
    /// use harp::shape::tracker::ShapeTracker;
    ///
    /// let graph_arc = Arc::new(Mutex::new(Graph::new()));
    /// let shape = ShapeTracker::new(vec![2, 3]);
    /// let input_tensor = Graph::new_input(graph_arc.clone(), shape);
    ///
    /// assert_eq!(graph_arc.lock().unwrap().inputs.len(), 1);
    /// ```
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

    /// Adds a new node to the graph.
    ///
    /// # Arguments
    ///
    /// * `node` - The `Node` to add.
    ///
    /// # Returns
    ///
    /// The `NodeIndex` of the newly added node.
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Adds an edge between two nodes in the graph.
    ///
    /// # Arguments
    ///
    /// * `from` - The `NodeIndex` of the source node.
    /// * `to` - The `NodeIndex` of the destination node.
    /// * `arg_index` - The argument index for the edge, indicating which input of the `to` node this edge represents.
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, arg_index: usize) {
        self.graph.add_edge(from, to, (arg_index, 0));
    }

    /// Registers a tensor as an output of the graph.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to the `Tensor` to register as an output.
    pub fn add_output(&mut self, tensor: &Tensor) {
        self.outputs.push(tensor.node_index);
    }

    // --- Accessors for testing and debugging ---

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns a reference to the node weight at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The `NodeIndex` of the node.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the `Node` if found, otherwise `None`.
    pub fn node_weight(&self, index: NodeIndex) -> Option<&Node> {
        self.graph.node_weight(index)
    }

    /// Returns an iterator over the parents of a given node.
    ///
    /// Each item in the iterator is a tuple containing the parent's `NodeIndex`
    /// and the argument index of the edge connecting it to the current node.
    ///
    /// # Arguments
    ///
    /// * `index` - The `NodeIndex` of the child node.
    pub fn parents(
        &self,
        index: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, (usize, usize))> + '_ {
        self.graph
            .edges_directed(index, Direction::Incoming)
            .map(|edge| (edge.source(), *edge.weight()))
    }

    /// Generates a DOT language representation of the graph.
    ///
    /// This string can be used with Graphviz tools to visualize the computation graph.
    /// Input nodes are filled with light gray, and output nodes have a double periphery.
    ///
    /// # Returns
    ///
    /// A `String` containing the DOT representation of the graph.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph {\n");

        // Nodes
        for (id, node) in self.graph.node_indices().map(|i| (i, &self.graph[i])) {
            let mut temp_shape = node.shape.clone();
            for (i, expr) in temp_shape.map.iter_mut().enumerate() {
                *expr = expr
                    .clone()
                    .replace(&Expr::Index, &Expr::Var(format!("idx{i}")));
            }
            for (i, expr) in temp_shape.max.iter_mut().enumerate() {
                *expr = expr
                    .clone()
                    .replace(&Expr::Index, &Expr::Var(format!("idx{i}")));
            }

            let label = format!("{:?}\n{}", node.op(), temp_shape);
            let mut attrs = vec![format!("label = \"{}\"", label)];

            if self.outputs.contains(&id) {
                attrs.push("peripheries=2".to_string());
            } else if self.inputs.contains(&id) {
                attrs.push("style=filled".to_string());
                attrs.push("fillcolor=lightgray".to_string());
            }

            dot.push_str(&format!("    {}[{}]\n", id.index(), attrs.join(", ")));
        }

        // Edges
        for edge in self.graph.edge_references() {
            let source = edge.source().index();
            let target = edge.target().index();
            let weight = edge.weight();
            let label = format!("({}, {})", weight.0, weight.1);
            dot.push_str(&format!(
                "    {source} -> {target} [label = \"{label}\"]\n"
            ));
        }

        dot.push('}');
        dot
    }

    /// Placeholder for graph optimization passes.
    ///
    /// This method is intended to implement various optimization techniques
    /// such as dead code elimination, constant folding, or operator fusion.
    pub fn optimize(&mut self) {
        // Implement optimization passes here, such as:
        // - Dead code elimination
        // - Constant folding
        // - Operator fusion
    }
}
