use crate::{graph::Graph, interpreter::Interpreter, node::Node, operator, tensor::TensorData};
use petgraph::{Direction, algo::toposort, graph::NodeIndex, visit::EdgeRef};
use std::collections::HashMap;

/// Trait for graph optimization passes.
///
/// Any struct implementing this trait can be used to optimize a computation graph.
pub trait GraphOptimizer {
    /// Applies an optimization pass to the given graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - A mutable reference to the `Graph` to be optimized.
    fn optimize(&mut self, graph: &mut Graph);
}

/// An optimizer that eliminates nodes from the graph that have no outgoing edges
/// and are not designated as output or input nodes.
pub struct EliminateUnusedNodes {}

impl GraphOptimizer for EliminateUnusedNodes {
    fn optimize(&mut self, graph: &mut Graph) {
        let mut nodes_to_remove = Vec::new();
        // Perform a topological sort to process nodes in a valid order.
        let topo_order = toposort(&graph.graph, None).unwrap();

        // Iterate in reverse topological order to identify unused nodes.
        for node_idx in topo_order.iter().rev() {
            // A node is considered unused if it has no outgoing edges (no consumers)
            // and is not explicitly marked as an output or input of the graph.
            if graph
                .graph
                .edges_directed(*node_idx, Direction::Outgoing)
                .count()
                == 0
                && !graph.outputs.contains(node_idx)
                && !graph.inputs.contains(node_idx)
            {
                nodes_to_remove.push(*node_idx);
            }
        }

        // Remove identified unused nodes from the graph.
        for node_idx in nodes_to_remove {
            graph.graph.remove_node(node_idx);
        }
    }
}

/// An optimizer that performs constant folding.
///
/// This optimization identifies subgraphs where all inputs are constants
/// and replaces the subgraph with a single constant node representing the computed value.
pub struct ConstantFolding {}

impl GraphOptimizer for ConstantFolding {
    fn optimize(&mut self, graph: &mut Graph) {
        let topo_order = toposort(&graph.graph, None).unwrap();
        let mut interpreter = Interpreter::new();
        // Map to store the evaluated constant data for nodes that can be folded.
        let mut nodes_to_fold: HashMap<NodeIndex, TensorData> = HashMap::new();

        // First pass: Identify nodes that can be folded and evaluate their constant values.
        for node_idx in topo_order {
            let node = graph.node_weight(node_idx).unwrap();
            let op = node.op();

            // If it's already a Const node, no need to fold it again.
            if op.as_any().downcast_ref::<operator::Const>().is_some() {
                continue;
            }

            // Check if all parents are constants.
            let mut all_parents_are_const = true;
            // Data for the interpreter to evaluate the node.
            let mut parent_eval_data = HashMap::<NodeIndex, TensorData>::new();

            let incoming_edges: Vec<(NodeIndex, usize)> = graph
                .graph
                .edges_directed(node_idx, Direction::Incoming)
                .map(|edge| (edge.source(), *edge.weight()))
                .collect();

            // If a node has no incoming edges and is not an Input, it cannot be evaluated by interpreter.
            if incoming_edges.is_empty() && op.as_any().downcast_ref::<operator::Input>().is_none() {
                all_parents_are_const = false;
            }

            for (parent_idx, _arg_idx) in &incoming_edges {
                let parent_node = graph.node_weight(*parent_idx).unwrap();
                if let Some(const_op) = parent_node.op().as_any().downcast_ref::<operator::Const>() {
                    parent_eval_data.insert(*parent_idx, const_op.data.clone());
                } else {
                    all_parents_are_const = false;
                    break;
                }
            }

            if all_parents_are_const {
                // If all parents are constants, evaluate this node.
                if let Ok(result_data) = interpreter.evaluate(node_idx, &graph.graph, &parent_eval_data) {
                    nodes_to_fold.insert(node_idx, result_data);
                }
            }
        }

        // Second pass: Perform the actual graph transformation (replace nodes).
        let mut old_to_new_node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut edges_to_add: Vec<(NodeIndex, NodeIndex, usize)> = Vec::new();
        let mut edges_to_remove: Vec<(NodeIndex, NodeIndex)> = Vec::new();

        for (old_node_idx, new_const_data) in nodes_to_fold {
            let old_node_data = graph.node_weight(old_node_idx).unwrap();
            let new_op = operator::Const { data: new_const_data };
            let new_node_data = Node::new(new_op, old_node_data.shape.clone());
            let new_node_idx = graph.graph.add_node(new_node_data);
            old_to_new_node_map.insert(old_node_idx, new_node_idx);

            // Collect outgoing edges from the old node to re-point them to the new constant node.
            for edge in graph.graph.edges_directed(old_node_idx, Direction::Outgoing) {
                edges_to_add.push((new_node_idx, edge.target(), *edge.weight()));
                edges_to_remove.push((old_node_idx, edge.target()));
            }

            // Collect incoming edges to the old node for removal.
            for edge in graph.graph.edges_directed(old_node_idx, Direction::Incoming) {
                edges_to_remove.push((edge.source(), old_node_idx));
            }

            // Update output list if the old node was an output.
            if let Some(pos) = graph.outputs.iter().position(|&x| x == old_node_idx) {
                graph.outputs[pos] = new_node_idx;
            }

            // Update input list if the old node was an input (unlikely for folding, but good to be safe).
            if let Some(pos) = graph.inputs.iter().position(|&x| x == old_node_idx) {
                graph.inputs[pos] = new_node_idx;
            }
        }

        // Perform edge removals and additions.
        // Note: petgraph::remove_edge requires EdgeIndex, so we iterate and remove.
        for (source, target) in edges_to_remove {
            if let Some(edge_idx) = graph.graph.find_edge(source, target) {
                graph.graph.remove_edge(edge_idx);
            }
        }

        for (source, target, weight) in edges_to_add {
            graph.graph.add_edge(source, target, weight);
        }

        // Remove the old nodes that have been replaced by new constant nodes.
        for (old_node_idx, _) in old_to_new_node_map {
            graph.graph.remove_node(old_node_idx);
        }
    }
}

/// A pipeline of multiple graph optimizers.
///
/// This struct allows applying a sequence of `GraphOptimizer`s to a graph
/// iteratively until no more changes occur or a maximum number of iterations is reached.
pub struct OptimizerPipeline {
    /// The list of optimizers to apply in sequence.
    optimizers: Vec<Box<dyn GraphOptimizer>>,
    /// The maximum number of iterations to run the optimization pipeline.
    max_iterations: usize,
}

impl OptimizerPipeline {
    /// Creates a new `OptimizerPipeline`.
    ///
    /// # Arguments
    ///
    /// * `optimizers` - A vector of boxed `GraphOptimizer` trait objects.
    /// * `max_iterations` - The maximum number of times to run through the optimizer list.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::optimizer::{OptimizerPipeline, EliminateUnusedNodes, ConstantFolding};
    ///
    /// let pipeline = OptimizerPipeline::new(
    ///     vec![
    ///         Box::new(EliminateUnusedNodes{}),
    ///         Box::new(ConstantFolding{}),
    ///     ],
    ///     10,
    /// );
    /// // Then you can use pipeline.optimize(graph);
    /// ```
    pub fn new(optimizers: Vec<Box<dyn GraphOptimizer>>, max_iterations: usize) -> Self {
        Self { optimizers, max_iterations }
    }
}

impl GraphOptimizer for OptimizerPipeline {
    fn optimize(&mut self, graph: &mut Graph) {
        let mut iteration = 0;
        loop {
            iteration += 1;
            let mut changed = false;
            // Capture the graph's state before applying optimizers in this iteration.
            let initial_graph_dot = graph.to_dot();

            // Apply each optimizer in the pipeline.
            for optimizer in self.optimizers.iter_mut() {
                optimizer.optimize(graph);
                // Check if the graph has changed after applying the current optimizer.
                if graph.to_dot() != initial_graph_dot { // This check is simplistic; a more robust check would compare graph structure, not just DOT string.
                    changed = true;
                }
            }

            // Break if no changes occurred in this iteration or max iterations reached.
            if !changed || iteration >= self.max_iterations {
                break;
            }
        }
    }
}