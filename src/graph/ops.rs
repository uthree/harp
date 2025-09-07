use crate::graph::{shape::view::View, ElementwiseOp, GraphNode, GraphNodeData, GraphOp, ReduceOp};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
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
                if self.shape() != rhs.shape() {
                    panic!(
                        "Mismatched shapes: expected {:?}, found {:?}",
                        self.shape(),
                        rhs.shape()
                    );
                }
                GraphNode(Rc::new(GraphNodeData {
                    op: GraphOp::Elementwise($op),
                    src: vec![self.clone(), rhs.clone()],
                    dtype: self.dtype.clone(),
                    view: self.view.clone(),
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

// Subtraction: a - b = a + (-b)
impl<'b> Sub<&'b GraphNode> for &GraphNode {
    type Output = GraphNode;
    fn sub(self, rhs: &'b GraphNode) -> Self::Output {
        self + &(-rhs)
    }
}

impl Sub for GraphNode {
    type Output = GraphNode;
    fn sub(self, rhs: Self) -> Self::Output {
        &self + &(-&rhs)
    }
}

// Division: a / b = a * (1/b)
impl<'b> Div<&'b GraphNode> for &GraphNode {
    type Output = GraphNode;
    fn div(self, rhs: &'b GraphNode) -> Self::Output {
        self * &rhs.recip()
    }
}

impl Div for GraphNode {
    type Output = GraphNode;
    fn div(self, rhs: Self) -> Self::Output {
        &self * &rhs.recip()
    }
}

impl Neg for &GraphNode {
    type Output = GraphNode;
    fn neg(self) -> Self::Output {
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Elementwise(ElementwiseOp::Neg),
            src: vec![self.clone()],
            dtype: self.dtype.clone(),
            view: self.view.clone(),
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
                view: self.view.clone(),
            }))
        }
    };
}

macro_rules! impl_graph_node_reduce_op {
    ($fname:ident, $op:expr) => {
        pub fn $fname(&self, axis: usize) -> GraphNode {
            assert!(
                axis < self.shape().len(),
                "Reduction axis is out of bounds."
            );
            let mut new_shape = self.shape().to_vec();
            new_shape.remove(axis);
            GraphNode(Rc::new(GraphNodeData {
                op: GraphOp::Reduce($op, axis),
                src: vec![self.clone()],
                dtype: self.dtype.clone(),
                view: View::new_contiguous(new_shape),
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
    impl_graph_node_reduce_op!(max_reduce, ReduceOp::Max);

    pub fn max(&self, rhs: &Self) -> Self {
        if self.dtype != rhs.dtype {
            panic!(
                "Mismatched dtypes: expected {:?}, found {:?}",
                self.dtype, rhs.dtype
            );
        }
        if self.shape() != rhs.shape() {
            panic!(
                "Mismatched shapes: expected {:?}, found {:?}",
                self.shape(),
                rhs.shape()
            );
        }
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Elementwise(ElementwiseOp::Max),
            src: vec![self.clone(), rhs.clone()],
            dtype: self.dtype.clone(),
            view: self.view.clone(),
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
        assert_eq!(c.src.len(), 2);
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
        assert_eq!(b.src.len(), 1);
    }

    #[test]
    fn test_derived_ops() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10)];
        let a = graph.input(DType::F32, shape.clone());
        let b = graph.input(DType::F32, shape.clone());

        // Test Sub: a - b -> Add(a, Neg(b))
        let c = &a - &b;
        assert!(matches!(c.op, GraphOp::Elementwise(ElementwiseOp::Add)));
        assert_eq!(c.src.len(), 2);
        assert!(Rc::ptr_eq(&c.src[0].0, &a.0));
        let neg_b = &c.src[1];
        assert!(matches!(neg_b.op, GraphOp::Elementwise(ElementwiseOp::Neg)));
        assert_eq!(neg_b.src.len(), 1);
        assert!(Rc::ptr_eq(&neg_b.src[0].0, &b.0));

        // Test Div: a / b -> Mul(a, Recip(b))
        let d = &a / &b;
        assert!(matches!(d.op, GraphOp::Elementwise(ElementwiseOp::Mul)));
        assert_eq!(d.src.len(), 2);
        assert!(Rc::ptr_eq(&d.src[0].0, &a.0));
        let recip_b = &d.src[1];
        assert!(matches!(
            recip_b.op,
            GraphOp::Elementwise(ElementwiseOp::Recip)
        ));
        assert_eq!(recip_b.src.len(), 1);
        assert!(Rc::ptr_eq(&recip_b.src[0].0, &b.0));
    }

    #[test]
    fn test_reduce_ops() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10), Expr::from(20)];
        let a = graph.input(DType::F32, shape.clone());

        let sum_node = a.sum(1);
        assert!(matches!(sum_node.op, GraphOp::Reduce(ReduceOp::Add, 1)));
        assert_eq!(sum_node.src.len(), 1);
        assert!(Rc::ptr_eq(&sum_node.src[0].0, &a.0));
        assert_eq!(sum_node.shape(), &[Expr::from(10)]);
    }

    #[test]
    #[should_panic(expected = "Reduction axis is out of bounds.")]
    fn test_reduce_ops_axis_out_of_bounds() {
        let mut graph = Graph::new();
        let shape = vec![Expr::from(10), Expr::from(20)];
        let a = graph.input(DType::F32, shape.clone());
        a.sum(2);
    }
}
