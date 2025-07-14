//! Provides the high-level `Tensor` API.

use harp_ir::{ComputationGraph, Dim, Graph, NodeId, Operator, Shape, ShapeTracker};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Add;

// --- High-Level Representation ---

#[derive(Debug, Clone)]
enum TensorOp {
    Load { name: String },
    Add,
}

type TensorGraph = Graph<TensorOp, usize>;

// --- Context and Tensor ---

pub struct Context {
    tensor_graph: RefCell<TensorGraph>,
    tracker_arena: RefCell<Vec<ShapeTracker>>,
}

#[derive(Clone, Copy)]
pub struct Tensor<'ctx> {
    ctx: &'ctx Context,
    node_id: NodeId,
    tracker_id: usize,
}

impl<'ctx> Tensor<'ctx> {
    pub fn shape(&self) -> Shape {
        self.ctx.tracker_arena.borrow()[self.tracker_id].shape.clone()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            tensor_graph: RefCell::new(TensorGraph::new()),
            tracker_arena: RefCell::new(Vec::new()),
        }
    }

    pub fn load(&self, name: &str, shape: Shape) -> Tensor<'_> {
        let tracker = ShapeTracker::new(shape);
        let tracker_id = self.alloc_tracker(tracker);
        let node_id = self.tensor_graph.borrow_mut().add_node(TensorOp::Load {
            name: name.to_string(),
        });
        Tensor { ctx: self, node_id, tracker_id }
    }

    fn alloc_tracker(&self, tracker: ShapeTracker) -> usize {
        let mut arena = self.tracker_arena.borrow_mut();
        arena.push(tracker);
        arena.len() - 1
    }
}

// --- Operator Overloading ---

impl<'ctx> Add for Tensor<'ctx> {
    type Output = Tensor<'ctx>;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_shape = self.shape();
        let rhs_shape = rhs.shape();
        assert_eq!(lhs_shape, rhs_shape, "Shapes must match for addition");

        let out_tracker = ShapeTracker::new(lhs_shape);
        let out_tracker_id = self.ctx.alloc_tracker(out_tracker);

        let mut graph = self.ctx.tensor_graph.borrow_mut();
        let add_node_id = graph.add_node(TensorOp::Add);
        graph.add_edge(add_node_id, self.node_id, 0);
        graph.add_edge(add_node_id, rhs.node_id, 1);

        Tensor {
            ctx: self.ctx,
            node_id: add_node_id,
            tracker_id: out_tracker_id,
        }
    }
}

// --- Lowering ---

type LoweringMap = HashMap<(NodeId, usize), NodeId>;

impl Context {
    pub fn lower(&self, outputs: &[Tensor]) -> (ComputationGraph, Vec<NodeId>) {
        let mut ir_graph = ComputationGraph::new();
        let mut high_to_low_map: LoweringMap = HashMap::new();
        let tensor_graph = self.tensor_graph.borrow();

        let low_outputs = outputs
            .iter()
            .map(|&high_tensor| {
                self.lower_recursive(high_tensor, &tensor_graph, &mut ir_graph, &mut high_to_low_map)
            })
            .collect();

        (ir_graph, low_outputs)
    }

    fn lower_recursive(
        &self,
        high_tensor: Tensor,
        tensor_graph: &TensorGraph,
        ir_graph: &mut ComputationGraph,
        high_to_low_map: &mut LoweringMap,
    ) -> NodeId {
        let memo_key = (high_tensor.node_id, high_tensor.tracker_id);
        if let Some(low_node_id) = high_to_low_map.get(&memo_key) {
            return *low_node_id;
        }

        let high_node = tensor_graph.get(high_tensor.node_id).unwrap();

        let low_children = high_node
            .children
            .iter()
            .map(|(edge, child_node_id)| {
                // This is the tricky part. Which tracker do we use for the child?
                // For now, we assume the child's "default" tracker (the one it was created with).
                // This requires a way to find the original tracker for a node.
                // This design is still flawed. Let's simplify.
                // We assume that for an operation, the inputs use their provided trackers.
                // This seems to be the case.
                // Let's find the Tensors corresponding to the children. This is not directly possible.
                
                // The lowering needs to be driven by the `outputs` list, and for each op,
                // we need to know the input tensors.
                // The graph `children` only gives us node IDs, not the `Tensor` handles.
                
                // Let's rethink the `Add` implementation. It should store the input `Tensor`s.
                // No, that would lead to cycles.
                
                // The current `lower_recursive` is fundamentally flawed because it loses
                // the tracker information for children.
                
                // Let's pass the full `Tensor` handles of the children to `lower_recursive`.
                // This is not possible with the current graph structure.
                
                // The information must flow from the output tensors downwards.
                // When we are at an `Add` node, we need to know which `Tensor`s were its inputs.
                // The graph only stores `NodeId`.
                
                // What if `TensorOp::Add` stored the tracker_ids of its inputs?
                // Let's try that.
                unimplemented!();
            })
            .collect::<Vec<_>>();

        let tracker = &self.tracker_arena.borrow()[high_tensor.tracker_id];
        let low_op = match &high_node.data {
            TensorOp::Load { name } => Operator::Load {
                name: name.clone(),
                shape: tracker.shape.clone(),
            },
            TensorOp::Add => Operator::Add {
                shape: tracker.shape.clone(),
            },
        };
        let low_node_id = ir_graph.add_node(low_op);

        // for (edge, low_child_id) in low_children {
        //     ir_graph.add_edge(low_node_id, low_child_id, edge);
        // }

        high_to_low_map.insert(memo_key, low_node_id);
        low_node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let ctx = Context::new();
        let shape = Shape::new(vec![Dim::Fixed(10)]);
        let a = ctx.load("a", shape.clone());
        let b = ctx.load("b", shape.clone());
        let c = a + b;
        assert_eq!(c.shape(), shape);
    }
}
