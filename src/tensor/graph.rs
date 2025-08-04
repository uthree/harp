use crate::ast::{AstNode, DType, Op as AstOp};
use crate::tensor::shape::expr::Expr;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

// Graph structure that owns all the nodes using interior mutability
#[derive(Default, Debug)]
pub struct Graph {
    pub nodes: RefCell<Vec<NodeData>>,
}

// A handle to a node in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

// A temporary view of a tensor in the graph
#[derive(Debug, Clone, Copy)]
pub struct NodeView<'a> {
    pub id: NodeId,
    pub graph: &'a Graph,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeData {
    pub op: TensorOp,
    pub src: Vec<NodeId>,
    pub dtype: DType,
    pub shape: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    Input,
    Elementwise(AstOp),
    Reduce(AstOp, usize),
    Contiguous,
    Permute(Vec<usize>),
    Concatenate(usize),

    // fused operators
    FusedElementwise(AstNode),
    FusedReduce(AstOp, Vec<usize>),
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: RefCell::new(Vec::new()),
        }
    }

    fn add_node(&self, op: TensorOp, src: Vec<NodeId>, dtype: DType, shape: Vec<Expr>) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len();
        nodes.push(NodeData {
            op,
            src,
            dtype,
            shape,
        });
        NodeId(id)
    }

    pub fn input(&self, dtype: DType, shape: Vec<Expr>) -> NodeView {
        let id = self.add_node(TensorOp::Input, vec![], dtype, shape);
        self.get_view(id)
    }

    pub fn get_view(&self, id: NodeId) -> NodeView {
        NodeView { id, graph: self }
    }

    pub fn add(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, lhs_shape, rhs_shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
                rhs_node.shape.clone(),
            )
        };
        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in add: {lhs_shape:?} vs {rhs_shape:?}");
        }
        let ast_node = AstNode::capture(0, lhs_dtype) + AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Add),
            vec![lhs, rhs],
            ast_node.dtype,
            lhs_shape,
        )
    }

    pub fn sub(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            ) // TODO: Proper shape calculation
        };
        let ast_node = AstNode::capture(0, lhs_dtype) - AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Add),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    pub fn mul(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            ) // TODO: Proper shape calculation
        };
        let ast_node = AstNode::capture(0, lhs_dtype) * AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Mul),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    pub fn div(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            ) // TODO: Proper shape calculation
        };
        let ast_node = AstNode::capture(0, lhs_dtype) / AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Mul),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    pub fn neg(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let ast_node = -AstNode::capture(0, dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Neg),
            vec![src],
            ast_node.dtype,
            shape,
        )
    }

    fn _reduce(&self, op: AstOp, src: NodeId, axis: usize) -> NodeId {
        let (dtype, mut shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        assert!(axis < shape.len(), "Reduction axis out of bounds");
        shape.remove(axis);
        self.add_node(TensorOp::Reduce(op, axis), vec![src], dtype, shape)
    }

    pub fn sum(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Add, src, axis)
    }

    pub fn max(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Max, src, axis)
    }

    pub fn prod(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Mul, src, axis)
    }
}

impl<'a> NodeView<'a> {
    pub fn op(&self) -> TensorOp {
        self.graph.nodes.borrow()[self.id.0].op.clone()
    }
    pub fn src(&self) -> Vec<NodeId> {
        self.graph.nodes.borrow()[self.id.0].src.clone()
    }
    pub fn dtype(&self) -> DType {
        self.graph.nodes.borrow()[self.id.0].dtype.clone()
    }
    pub fn shape(&self) -> Vec<Expr> {
        self.graph.nodes.borrow()[self.id.0].shape.clone()
    }

    pub fn sum(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.sum(self.id, axis);
        self.graph.get_view(new_id)
    }

    pub fn max(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.max(self.id, axis);
        self.graph.get_view(new_id)
    }

    pub fn prod(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.prod(self.id, axis);
        self.graph.get_view(new_id)
    }
}

// --- Operator Overloads for NodeView ---

impl<'a> Add for NodeView<'a> {
    type Output = NodeView<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.add(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Sub for NodeView<'a> {
    type Output = NodeView<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.sub(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Mul for NodeView<'a> {
    type Output = NodeView<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.mul(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Div for NodeView<'a> {
    type Output = NodeView<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.div(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Neg for NodeView<'a> {
    type Output = NodeView<'a>;
    fn neg(self) -> Self::Output {
        let new_id = self.graph.neg(self.id);
        self.graph.get_view(new_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, Op as AstOp};

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
        assert_eq!(c.op(), TensorOp::Elementwise(AstOp::Add));
        assert_eq!(c.src(), vec![a.id, b.id]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_view_neg() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = -a;

        assert_eq!(b.id.0, 1);
        assert_eq!(b.op(), TensorOp::Elementwise(AstOp::Neg));
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

        assert_eq!(d.op(), TensorOp::Elementwise(AstOp::Add));
        let mul_node = d.graph.get_view(d.src()[0]);
        assert_eq!(mul_node.op(), TensorOp::Elementwise(AstOp::Mul));
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

        assert_eq!(b.op(), TensorOp::Reduce(AstOp::Add, 1));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![10.into(), 30.into()]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_reduce_max() {
        let graph = Graph::new();
        let a = graph.input(DType::I32, vec![10.into(), 20.into()]);
        let b = a.max(0);

        assert_eq!(b.op(), TensorOp::Reduce(AstOp::Max, 0));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![20.into()]);
        assert_eq!(b.dtype(), DType::I32);
    }

    #[test]
    fn test_reduce_prod() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into(), 30.into()]);
        let b = a.prod(1);

        assert_eq!(b.op(), TensorOp::Reduce(AstOp::Mul, 1));
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
}
