//! Provides the high-level `Tensor` API.
//!
//! This crate allows users to build a high-level computation graph, which can then
//! be lowered into the `harp-ir` representation for compilation.

use harp_ir::{ComputationGraph, Graph, NodeId, Operator};
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
    ///
    /// It takes a slice of `Tensor`s as the desired outputs and returns the
    /// `ComputationGraph` along with the corresponding `NodeId`s for those outputs.
    pub fn lower(&self, outputs: &[Tensor]) -> (ComputationGraph, Vec<NodeId>) {
        let mut ir_graph = ComputationGraph::new();
        let mut high_to_low_map = HashMap::new();
        let tensor_graph = self.tensor_graph.borrow();

        let low_outputs = outputs
            .iter()
            .map(|&high_tensor| {
                self.lower_recursive(
                    high_tensor.node_id,
                    &tensor_graph,
                    &mut ir_graph,
                    &mut high_to_low_map,
                )
            })
            .collect();

        (ir_graph, low_outputs)
    }

    /// Helper function to recursively lower the graph.
    fn lower_recursive(
        &self,
        high_node_id: NodeId,
        tensor_graph: &TensorGraph,
        ir_graph: &mut ComputationGraph,
        high_to_low_map: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(low_node_id) = high_to_low_map.get(&high_node_id) {
            return *low_node_id;
        }

        let high_node = tensor_graph.get(high_node_id).unwrap();

        let mut low_children = Vec::new();
        for (edge, high_child_id) in &high_node.children {
            let low_child_id =
                self.lower_recursive(*high_child_id, tensor_graph, ir_graph, high_to_low_map);
            low_children.push((*edge, low_child_id));
        }

        let low_op = match &high_node.data {
            HighLevelOp::Load { name } => Operator::Load { name: name.clone() },
            HighLevelOp::Add => Operator::Add,
        };
        let low_node_id = ir_graph.add_node(low_op);

        for (edge, low_child_id) in low_children {
            ir_graph.add_edge(low_node_id, low_child_id, edge);
        }

        high_to_low_map.insert(high_node_id, low_node_id);
        low_node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowering_single_output() {
        let ctx = Context::new();
        let a = ctx.load("a");
        let b = ctx.load("b");
        let c = a + b;

        let (ir_graph, low_outputs) = ctx.lower(&[c]);

        assert_eq!(low_outputs.len(), 1);
        
        let dot = ir_graph.to_dot();
        // The final node in the lowered graph should be the output.
        let expected_output_id = ir_graph.len() - 1;
        assert_eq!(low_outputs[0], NodeId::from(expected_output_id));

        let expected_dot = r#"digraph G {
  node [shape=box];
  n0 [label="Load(a)"];
  n1 [label="Load(b)"];
  n2 [label="Add"];
  n2 -> n0 [label="0"];
  n2 -> n1 [label="1"];
}
"#;
        assert_eq!(dot, expected_dot);
    }

    #[test]
    fn test_lowering_multiple_outputs() {
        let ctx = Context::new();
        let a = ctx.load("a");
        let b = ctx.load("b");
        let c = a + b;
        let d = a + c;

        // Lower with `c` and `d` as outputs.
        let (ir_graph, low_outputs) = ctx.lower(&[c, d]);

        assert_eq!(low_outputs.len(), 2);

        // We expect 4 nodes in the IR graph: a, b, (a+b), (a+(a+b))
        assert_eq!(ir_graph.len(), 4);
    }
}
