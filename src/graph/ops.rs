use crate::graph::{ElementwiseOp, GraphNode, GraphNodeData, GraphOp, ReduceOp};
use std::ops::{Add, Mul, Neg, Rem};
use std::rc::Rc;

macro_rules! impl_graph_node_binary_op {
    ($trait:ident, $fname:ident, $op:expr) => {
        impl<'a, 'b> $trait<&'b GraphNode> for &'a GraphNode {
            type Output = GraphNode;
            fn $fname(self, rhs: &'b GraphNode) -> Self::Output {
                if self.dtype != rhs.dtype {
                    panic!(
                        "Mismatched dtypes: expected {:?}, found {:?}",
                        self.dtype, rhs.dtype
                    );
                }
                if self.shape != rhs.shape {
                    panic!(
                        "Mismatched shapes: expected {:?}, found {:?}",
                        self.shape, rhs.shape
                    );
                }
                GraphNode(Rc::new(GraphNodeData {
                    op: GraphOp::Elementwise($op),
                    src: vec![self.clone(), rhs.clone()],
                    dtype: self.dtype.clone(),
                    shape: self.shape.clone(),
                }))
            }
        }

        impl $trait for GraphNode {
            type Output = GraphNode;
            fn $fname(self, rhs: Self) -> Self::Output {
                (&self).$fname(&rhs)
            }
        }
    };
}

impl_graph_node_binary_op!(Add, add, ElementwiseOp::Add);
impl_graph_node_binary_op!(Mul, mul, ElementwiseOp::Mul);
impl_graph_node_binary_op!(Rem, rem, ElementwiseOp::Rem);

impl Neg for &GraphNode {
    type Output = GraphNode;
    fn neg(self) -> Self::Output {
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Elementwise(ElementwiseOp::Neg),
            src: vec![self.clone()],
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
        }))
    }
}

impl Neg for GraphNode {
    type Output = GraphNode;
    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}

macro_rules! impl_graph_node_unary_op {
    ($fname:ident, $op:expr) => {
        pub fn $fname(&self) -> GraphNode {
            GraphNode(Rc::new(GraphNodeData {
                op: GraphOp::Elementwise($op),
                src: vec![self.clone()],
                dtype: self.dtype.clone(),
                shape: self.shape.clone(),
            }))
        }
    };
}

macro_rules! impl_graph_node_reduce_op {
    ($fname:ident, $op:expr) => {
        pub fn $fname(&self, axis: usize) -> GraphNode {
            assert!(axis < self.shape.len(), "Reduction axis is out of bounds.");
            let mut new_shape = self.shape.clone();
            new_shape.remove(axis);
            GraphNode(Rc::new(GraphNodeData {
                op: GraphOp::Reduce($op, axis),
                src: vec![self.clone()],
                dtype: self.dtype.clone(),
                shape: new_shape,
            }))
        }
    };
}

impl GraphNode {
    impl_graph_node_unary_op!(recip, ElementwiseOp::Recip);
    impl_graph_node_unary_op!(sqrt, ElementwiseOp::Sqrt);
    impl_graph_node_unary_op!(sin, ElementwiseOp::Sin);
    impl_graph_node_unary_op!(log2, ElementwiseOp::Log2);
    impl_graph_node_unary_op!(exp2, ElementwiseOp::Exp2);

    impl_graph_node_reduce_op!(sum, ReduceOp::Add);
    impl_graph_node_reduce_op!(prod, ReduceOp::Mul);
    impl_graph_node_reduce_op!(max, ReduceOp::Max);

    pub fn max2(&self, rhs: &Self) -> Self {
        if self.dtype != rhs.dtype {
            panic!(
                "Mismatched dtypes: expected {:?}, found {:?}",
                self.dtype, rhs.dtype
            );
        }
        if self.shape != rhs.shape {
            panic!(
                "Mismatched shapes: expected {:?}, found {:?}",
                self.shape, rhs.shape
            );
        }
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Elementwise(ElementwiseOp::Max),
            src: vec![self.clone(), rhs.clone()],
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::Expr;
    use crate::graph::Graph;

    #[test]
    fn test_binary_ops_ok() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10)];
        let a = graph.input(DType::F32, shape.clone());
        let b = graph.input(DType::F32, shape.clone());

        let c = &a + &b;
        assert!(matches!(c.op, GraphOp::Elementwise(ElementwiseOp::Add)));

        let d = &a * &b;
        assert!(matches!(d.op, GraphOp::Elementwise(ElementwiseOp::Mul)));

        let e = &a % &b;
        assert!(matches!(e.op, GraphOp::Elementwise(ElementwiseOp::Rem)));

        let f = a.max2(&b);
        assert!(matches!(f.op, GraphOp::Elementwise(ElementwiseOp::Max)));
    }

    #[test]
    #[should_panic(expected = "Mismatched dtypes")]
    fn test_binary_ops_dtype_mismatch() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10)];
        let a = graph.input(DType::F32, shape.clone());
        let b = graph.input(DType::Isize, shape.clone());
        let _ = &a + &b;
    }

    #[test]
    #[should_panic(expected = "Mismatched shapes")]
    fn test_binary_ops_shape_mismatch() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(20)]);
        let _ = &a + &b;
    }

    #[test]
    fn test_unary_ops() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10)];
        let a = graph.input(DType::F32, shape.clone());

        let b = -&a;
        assert!(matches!(b.op, GraphOp::Elementwise(ElementwiseOp::Neg)));

        let c = a.recip();
        assert!(matches!(c.op, GraphOp::Elementwise(ElementwiseOp::Recip)));
    }

    #[test]
    fn test_reduce_ops() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10), Expr::from(20)];
        let a = graph.input(DType::F32, shape.clone());

        // Test sum
        let sum_node = a.sum(1);
        assert!(matches!(sum_node.op, GraphOp::Reduce(ReduceOp::Add, 1)));
        assert_eq!(sum_node.shape, vec![Expr::from(10)]);

        // Test prod
        let prod_node = a.prod(0);
        assert!(matches!(prod_node.op, GraphOp::Reduce(ReduceOp::Mul, 0)));
        assert_eq!(prod_node.shape, vec![Expr::from(20)]);

        // Test max_reduce
        let max_node = a.max(1);
        assert!(matches!(max_node.op, GraphOp::Reduce(ReduceOp::Max, 1)));
        assert_eq!(max_node.shape, vec![Expr::from(10)]);
    }

    #[test]
    #[should_panic(expected = "Reduction axis is out of bounds.")]
    fn test_reduce_ops_axis_out_of_bounds() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10), Expr::from(20)];
        let a = graph.input(DType::F32, shape.clone());
        a.sum(2); // Axis 2 is out of bounds for a 2D tensor
    }
}
