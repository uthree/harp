//! Provides the high-level `Tensor` API.

use harp_ir::{ComputationGraph, Dim, Graph, NodeId, Operator, Shape};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Add;

// --- High-Level Representation ---

#[derive(Debug, Clone)]
enum TensorOp {
    Load { name: String, shape: Shape },
    Add { shape: Shape },
}

impl TensorOp {
    fn shape(&self) -> &Shape {
        match self {
            TensorOp::Load { shape, .. } => shape,
            TensorOp::Add { shape } => shape,
        }
    }
}

type TensorGraph = Graph<TensorOp, usize>;

// --- Context and Tensor ---

pub struct Context {
    tensor_graph: RefCell<TensorGraph>,
}

#[derive(Clone, Copy)]
pub struct Tensor<'ctx> {
    ctx: &'ctx Context,
    node_id: NodeId,
}

impl<'ctx> Tensor<'ctx> {
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> Shape {
        self.ctx.tensor_graph.borrow().get(self.node_id).unwrap().data.shape().clone()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            tensor_graph: RefCell::new(TensorGraph::new()),
        }
    }

    pub fn load(&self, name: &str, shape: Shape) -> Tensor<'_> {
        let node_id = self.tensor_graph.borrow_mut().add_node(TensorOp::Load {
            name: name.to_string(),
            shape,
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
        let lhs_shape = self.shape();
        let rhs_shape = rhs.shape();

        // For now, require shapes to be identical for addition.
        // A real implementation would support broadcasting rules.
        assert_eq!(lhs_shape, rhs_shape, "Shapes must match for addition");

        let mut graph = self.ctx.tensor_graph.borrow_mut();
        let add_node_id = graph.add_node(TensorOp::Add { shape: lhs_shape });
        graph.add_edge(add_node_id, self.node_id, 0);
        graph.add_edge(add_node_id, rhs.node_id, 1);
        Tensor {
            ctx: self.ctx,
            node_id: add_node_id,
        }
    }
}

// --- Lowering ---

impl Context {
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
            TensorOp::Load { name, shape } => Operator::Load { name: name.clone(), shape: shape.clone() },
            TensorOp::Add { .. } => Operator::Add, // Shape info is not on the IR Add op yet
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
    fn test_shape_inference() {
        let ctx = Context::new();
        let shape = Shape::new(vec![Dim::Fixed(10), Dim::Symbolic("N".to_string())]);
        let a = ctx.load("a", shape.clone());
        let b = ctx.load("b", shape.clone());
        let c = a + b;

        assert_eq!(c.shape(), shape);
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_panic() {
        let ctx = Context::new();
        let shape1 = Shape::new(vec![Dim::Fixed(10)]);
        let shape2 = Shape::new(vec![Dim::Fixed(11)]);
        let a = ctx.load("a", shape1);
        let b = ctx.load("b", shape2);
        let _ = a + b; // This should panic
    }

    #[test]
    fn test_lowering_with_shapes() {
        let ctx = Context::new();
        let shape = Shape::new(vec![Dim::Fixed(10)]);
        let a = ctx.load("a", shape.clone());
        let b = ctx.load("b", shape.clone());
        let c = a + b;

        let (ir_graph, _) = ctx.lower(&[c]);
        
        let dot = ir_graph.to_dot();
        let expected_dot = r#"digraph G {
  node [shape=box];
  n0 [label="Load(a, [10])"];
  n1 [label="Load(b, [10])"];
  n2 [label="Add"];
  n2 -> n0 [label="0"];
  n2 -> n1 [label="1"];
}
"#;
        assert_eq!(dot, expected_dot);
    }
}
