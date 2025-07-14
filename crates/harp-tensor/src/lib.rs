//! Provides the high-level `Tensor` API.
//!
//! This crate allows users to build a high-level computation graph, which can then
//! be lowered into the `harp-ir` representation for compilation.

use harp_ir::{ComputationGraph, Operator};
use harp_graph::{Graph, NodeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Add;

// --- High-Level Representation ---

/// Operations in the high-level tensor graph.
#[derive(Debug, Clone)]
enum HighLevelOp {
    Load { name: String },
    Add,
}

/// The high-level graph built by tensor operations.
type TensorGraph = Graph<HighLevelOp, usize>;

// --- Context and Tensor ---

/// Owns and manages the high-level computation graph.
pub struct Context {
    tensor_graph: RefCell<TensorGraph>,
}

/// A lightweight handle to a node in the high-level computation graph.
#[derive(Clone, Copy)]
pub struct Tensor<'ctx> {
    ctx: &'ctx Context,
    node_id: NodeId,
}

impl Context {
    pub fn new() -> Self {
        Self {
            tensor_graph: RefCell::new(TensorGraph::new()),
        }
    }

    /// Creates a new tensor representing a load operation.
    pub fn load(&self, name: &str) -> Tensor<'_> {
        let node_id = self.tensor_graph.borrow_mut().add_node(HighLevelOp::Load {
            name: name.to_string(),
        });
        Tensor {
            ctx: self,
            node_id,
        }
    }
}

// --- Operator Overloading ---

impl<'ctx> Add for Tensor<'ctx> {
    type Output = Tensor<'ctx>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut graph = self.ctx.tensor_graph.borrow_mut();
        let add_node_id = graph.add_node(HighLevelOp::Add);
        graph.add_edge(add_node_id, self.node_id, 0);
        graph.add_edge(add_node_id, rhs.node_id, 1);
        Tensor {
            ctx: self.ctx,
            node_id: add_node_id,
        }
    }
}

// --- Lowering (High-Level IR -> Low-Level IR) ---

impl Context {
    /// Lowers the high-level tensor graph into the `harp-ir` `ComputationGraph`.
    pub fn lower(&self) -> ComputationGraph {
        let mut ir_graph = ComputationGraph::new();
        let mut high_to_low_map = HashMap::new();
        let tensor_graph = self.tensor_graph.borrow();

        // We need to find the output nodes of the graph to start the traversal.
        // For now, we assume the last node added is the output.
        // A more robust solution would be to track outputs explicitly.
        if let Some(output_node_id) = (0..tensor_graph.len()).last().map(NodeId::from) {
            self.lower_recursive(output_node_id, &tensor_graph, &mut ir_graph, &mut high_to_low_map);
        }

        ir_graph
    }

    /// Helper function to recursively lower the graph.
    fn lower_recursive(
        &self,
        high_node_id: NodeId,
        tensor_graph: &TensorGraph,
        ir_graph: &mut ComputationGraph,
        high_to_low_map: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // If we've already processed this node, return its ID in the new graph.
        if let Some(low_node_id) = high_to_low_map.get(&high_node_id) {
            return *low_node_id;
        }

        let high_node = tensor_graph.get(high_node_id).unwrap();

        // First, lower the children.
        let mut low_children = Vec::new();
        for (edge, high_child_id) in &high_node.children {
            let low_child_id = self.lower_recursive(*high_child_id, tensor_graph, ir_graph, high_to_low_map);
            low_children.push((*edge, low_child_id));
        }

        // Then, create the corresponding node in the low-level graph.
        let low_op = match &high_node.data {
            HighLevelOp::Load { name } => Operator::Load { name: name.clone() },
            HighLevelOp::Add => Operator::Add,
        };
        let low_node_id = ir_graph.add_node(low_op);

        // Add the edges to the new node.
        for (edge, low_child_id) in low_children {
            ir_graph.add_edge(low_node_id, low_child_id, edge);
        }

        // Memoize the result.
        high_to_low_map.insert(high_node_id, low_node_id);

        low_node_id
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowering() {
        let ctx = Context::new();
        let a = ctx.load("a");
        let b = ctx.load("b");
        let _c = a + b; // This creates the high-level graph.

        // Now, lower it to the IR graph.
        let ir_graph = ctx.lower();

        // The lowered graph should have the correct structure.
        // Note: The node order might be different due to the recursive lowering.
        // `to_dot` provides a canonical representation if the traversal is deterministic.
        let dot = ir_graph.to_dot();
        let expected = r#"digraph G {
  node [shape=box];
  n0 [label="Load(a)"];
  n1 [label="Load(b)"];
  n2 [label="Add"];
  n2 -> n0 [label="0"];
  n2 -> n1 [label="1"];
}
"#;
        // This assertion is brittle because node IDs depend on traversal order.
        // A better test would parse the DOT file or inspect the graph structure.
        // For now, we rely on the deterministic nature of our simple recursive lowering.
        assert_eq!(dot, expected);
    }
}