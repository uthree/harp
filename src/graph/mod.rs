//! Defines the core computation graph structure of the tensor library.
//!
//! This module provides `Graph`, `NodeId`, and `NodeView`, which are the fundamental
//! components for building and manipulating deferred computation graphs. Operations
//! on `NodeView`s construct a graph of `NodeData` nodes, which can then be
//! compiled and executed.

pub mod context;
pub mod lowerer;
pub mod node;
pub mod op;
pub mod ops;
pub mod shape;
pub mod view;

pub use context::Graph;
pub use node::{NodeData, NodeId};
pub use op::GraphOp;
pub use ops::{ConvolutionOps, ElementwiseOps, ReduceOps, ShapeOps};
pub use view::NodeView;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstOp, Const, DType};
    use crate::graph::shape::expr::Expr;

    #[test]
    fn test_graph_creation_and_view() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = graph.input(DType::F32, vec![]);
        assert_eq!(a.id.0, 0);
        assert_eq!(b.id.0, 1);
        assert_eq!(graph.nodes.borrow().len(), 2);
        assert_eq!(a.dtype(), DType::F32);
    }

    #[test]
    fn test_view_add() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = graph.input(DType::F32, vec![]);
        let c = a + b;

        assert_eq!(c.id.0, 2);
        assert_eq!(c.op(), GraphOp::Elementwise(AstOp::Add));
        assert_eq!(c.src(), vec![a.id, b.id]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_view_neg() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = -a;

        assert_eq!(b.id.0, 1);
        assert_eq!(b.op(), GraphOp::Elementwise(AstOp::Neg));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_view_implicit_cast() {
        let graph = Graph::new();
        let a = graph.input(DType::I32, vec![]);
        let b = graph.input(DType::F32, vec![]);
        let c = a + b;

        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_complex_expression_with_views() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = graph.input(DType::F32, vec![]);
        let c = graph.input(DType::F32, vec![]);
        // d = a * b + c
        let d = a * b + c;

        assert_eq!(d.op(), GraphOp::Elementwise(AstOp::Add));
        let mul_node = d.graph.get_view(d.src()[0]);
        assert_eq!(mul_node.op(), GraphOp::Elementwise(AstOp::Mul));
        assert_eq!(mul_node.src(), vec![a.id, b.id]);
        assert_eq!(d.src()[1], c.id);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch in add")]
    fn test_add_shape_mismatch_panics() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(20)]);
        let _ = a + b;
    }

    #[test]
    fn test_reduce_sum() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into(), 30.into()]);
        let b = a.sum(1);

        assert_eq!(b.op(), GraphOp::Reduce(AstOp::Add, 1));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![10.into(), 30.into()]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_reduce_max() {
        let graph = Graph::new();
        let a = graph.input(DType::I32, vec![10.into(), 20.into()]);
        let b = a.max(0);

        assert_eq!(b.op(), GraphOp::Reduce(AstOp::Max, 0));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![20.into()]);
        assert_eq!(b.dtype(), DType::I32);
    }

    #[test]
    fn test_reduce_prod() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into(), 30.into()]);
        let b = a.prod(1);

        assert_eq!(b.op(), GraphOp::Reduce(AstOp::Mul, 1));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![10.into(), 30.into()]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    #[should_panic(expected = "Reduction axis out of bounds")]
    fn test_reduce_axis_out_of_bounds() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into()]);
        a.sum(1);
    }

    #[test]
    fn test_input_registration() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = graph.input(DType::I32, vec![10.into()]);

        let inputs = graph.inputs.borrow();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], a.id);
        assert_eq!(inputs[1], b.id);
    }

    #[test]
    fn test_as_output() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = graph.input(DType::F32, vec![]);
        let c = (a + b).as_output();

        assert_eq!(graph.outputs.borrow().len(), 1);
        assert_eq!(graph.outputs.borrow()[0], c.id);
    }

    #[test]
    fn test_permute() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let b = a.permute(vec![1, 0]);

        assert_eq!(b.op(), GraphOp::Permute(vec![1, 0]));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![20.into(), 10.into()]);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 1.into(), 20.into()]);
        let b = a.squeeze(1);
        let c = b.unsqueeze(0);

        assert_eq!(b.op(), GraphOp::Squeeze(1));
        assert_eq!(b.shape(), vec![10.into(), 20.into()]);

        assert_eq!(c.op(), GraphOp::Unsqueeze(0));
        assert_eq!(c.shape(), vec![1.into(), 10.into(), 20.into()]);
    }

    #[test]
    fn test_expand() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![1.into(), 20.into()]);
        let new_shape = vec![10.into(), 20.into()];
        let b = a.expand(new_shape.clone());

        assert_eq!(b.op(), GraphOp::Expand(new_shape.clone()));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), new_shape);
    }

    #[test]
    #[should_panic]
    fn test_expand_invalid() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![2.into(), 20.into()]);
        let new_shape = vec![10.into(), 20.into()];
        // This should panic because the original dimension is not 1.
        a.expand(new_shape);
    }

    #[test]
    fn test_contiguous() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let b = a.contiguous();

        assert_eq!(b.op(), GraphOp::Contiguous);
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), a.shape());
    }

    #[test]
    fn test_graph_equality() {
        // Helper to create a simple graph: a + b
        let create_graph = || {
            let graph = Graph::new();
            let a = graph.input(DType::F32, vec![]);
            let b = graph.input(DType::F32, vec![]);
            (a + b).as_output();
            graph
        };

        let graph1 = create_graph();
        let graph2 = create_graph();

        assert_eq!(graph1, graph2);
    }

    #[test]
    fn test_graph_inequality() {
        // Graph 1: a + b
        let create_graph1 = || {
            let graph = Graph::new();
            let a = graph.input(DType::F32, vec![]);
            let b = graph.input(DType::F32, vec![]);
            (a + b).as_output();
            graph
        };

        // Graph 2: a * b
        let create_graph2 = || {
            let graph = Graph::new();
            let a = graph.input(DType::F32, vec![]);
            let b = graph.input(DType::F32, vec![]);
            (a * b).as_output();
            graph
        };

        let graph1 = create_graph1();
        let graph2 = create_graph2();

        assert_ne!(graph1, graph2);
    }

    #[test]
    fn test_cumulative_sum() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let b = a.cumsum(1);

        assert_eq!(b.op(), GraphOp::Cumulative(AstOp::Add, 1));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![10.into(), 20.into()]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_slice() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let b = a.slice(vec![(2.into(), 8.into()), (5.into(), 15.into())]);

        assert_eq!(
            b.op(),
            GraphOp::Slice(vec![(2.into(), 8.into()), (5.into(), 15.into())])
        );
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![6.into(), 10.into()]);
    }

    #[test]
    fn test_full() {
        let graph = Graph::new();
        let shape = vec![10.into(), 20.into()];
        let a = graph.full(42.0f32, shape.clone());

        assert_eq!(a.op(), GraphOp::Full(Const::F32(42.0)));
        assert_eq!(a.src().len(), 0);
        assert_eq!(a.shape(), shape);
        assert_eq!(a.dtype(), DType::F32);
    }

    #[test]
    fn test_rand() {
        let graph = Graph::new();
        let shape = vec![10.into(), 20.into()];
        let a = graph.rand(DType::F32, shape.clone());

        assert_eq!(a.op(), GraphOp::Rand);
        assert_eq!(a.src().len(), 0);
        assert_eq!(a.shape(), shape);
        assert_eq!(a.dtype(), DType::F32);
    }

    #[test]
    fn test_unfold1d() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![1.into(), 3.into(), 10.into()]);
        let b = a.unfold1d(2, 3, 1);

        assert_eq!(
            b.op(),
            GraphOp::Unfold1d {
                dim: 2,
                kernel_size: 3,
                stride: 1
            }
        );
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![1.into(), 3.into(), 8.into(), 3.into()]);
    }

    #[test]
    fn test_reshape() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let b = a.reshape(vec![200.into()]);
        let c = b.reshape(vec![20.into(), 10.into()]);

        assert_eq!(b.shape(), vec![200.into()]);
        assert_eq!(c.shape(), vec![20.into(), 10.into()]);

        // Check that contiguous was inserted
        let reshape_node = &graph.nodes.borrow()[b.id.0];
        let contiguous_node_op = &graph.nodes.borrow()[reshape_node.src[0].0].op;
        assert!(matches!(contiguous_node_op, GraphOp::Contiguous));
    }

    #[test]
    fn test_conv1d() {
        let graph = Graph::new();
        let groups = 2;
        let x = graph.input(DType::F32, vec![1.into(), (4 * groups).into(), 10.into()]); // N, C_in, L
        let w = graph.input(DType::F32, vec![(2 * groups).into(), 4.into(), 3.into()]); // C_out, C_in/G, K
        let y = x.conv1d(w, 3, 1, groups);

        // Expected output shape: [N, C_out, L_out] = [1, 4, 8]
        assert_eq!(y.shape(), vec![1.into(), (2 * groups).into(), 8.into()]);

        // Check the final operation is a Reshape
        assert!(matches!(y.op(), GraphOp::Reshape(_)));

        // Trace back the graph to check the structure
        let contiguous_node = graph.get_view(y.src()[0]);
        assert!(matches!(contiguous_node.op(), GraphOp::Contiguous));

        let sum_c_in = graph.get_view(contiguous_node.src()[0]);
        assert!(matches!(sum_c_in.op(), GraphOp::Reduce(AstOp::Add, 3)));

        let sum_k = graph.get_view(sum_c_in.src()[0]);
        assert!(matches!(sum_k.op(), GraphOp::Reduce(AstOp::Add, 5)));
    }
}
