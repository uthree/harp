//! Defines the core computation graph structure of the tensor library.
//!
//! This module provides `Graph`, `NodeId`, and `NodeView`, which are the fundamental
//! components for building and manipulating deferred computation graphs. Operations
//! on `NodeView`s construct a graph of `NodeData` nodes, which can then be
//! compiled and executed.

use crate::ast::{AstNode, DType, Op as AstOp};
use crate::tensor::shape::expr::Expr;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Owns all the nodes of a computation graph.
///
/// The `Graph` uses interior mutability (`RefCell`) to allow nodes to be added
/// dynamically while maintaining immutable references to the graph itself.
#[derive(Default, Debug)]
pub struct Graph {
    /// A vector holding the data for all nodes in the graph.
    pub nodes: RefCell<Vec<NodeData>>,
    /// A list of node IDs that are considered inputs to the graph.
    pub inputs: RefCell<Vec<NodeId>>,
    /// A list of node IDs that are considered outputs of the graph.
    pub outputs: RefCell<Vec<NodeId>>,
}

/// A unique identifier for a node within a `Graph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A temporary, lightweight handle to a node in the graph.
///
/// `NodeView` provides a convenient, chainable API for building the computation graph.
/// It holds a reference to the graph and the ID of the node it represents.
/// Most tensor operations are implemented on `NodeView`.
///
/// # Examples
///
/// ```
/// use harp::tensor::graph::Graph;
/// use harp::ast::DType;
///
/// let graph = Graph::new();
/// let a = graph.input(DType::F32, vec![]);
/// let b = graph.input(DType::F32, vec![]);
/// let c = a + b; // Creates a new node in the graph
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NodeView<'a> {
    pub id: NodeId,
    pub graph: &'a Graph,
}

/// The data associated with a single node in the computation graph.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeData {
    /// The operation performed by this node.
    pub op: TensorOp,
    /// The `NodeId`s of the input nodes to this operation.
    pub src: Vec<NodeId>,
    /// The data type of the tensor produced by this node.
    pub dtype: DType,
    /// The symbolic shape of the tensor.
    pub shape: Vec<Expr>,
}

/// An enumeration of all possible tensor operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    /// An input tensor to the graph.
    Input,
    /// An element-wise operation (e.g., add, mul, sin).
    Elementwise(AstOp),
    /// A reduction operation along a specific axis (e.g., sum, max).
    Reduce(AstOp, usize),
    /// An operation that makes the memory layout of a tensor contiguous.
    Contiguous,
    /// An operation that permutes the axes of a tensor.
    Permute(Vec<usize>),
    /// Removes a dimension of size 1.
    Squeeze(usize),
    /// Adds a dimension of size 1.
    Unsqueeze(usize),
    /// Expands a tensor to a new shape.
    Expand(Vec<Expr>),
    /// An operation that concatenates tensors along a specific axis.
    Concatenate(usize),

    // Fused operators for optimization
    /// A fused sequence of element-wise operations.
    FusedElementwise(AstNode),
    /// A fused reduction operation.
    FusedReduce(AstOp, Vec<usize>),
}

impl Graph {
    /// Creates a new, empty computation graph.
    pub fn new() -> Self {
        Graph {
            nodes: RefCell::new(Vec::new()),
            inputs: RefCell::new(Vec::new()),
            outputs: RefCell::new(Vec::new()),
        }
    }

    /// Adds a new node to the graph. This is an internal method.
    pub fn add_node(
        &self,
        op: TensorOp,
        src: Vec<NodeId>,
        dtype: DType,
        shape: Vec<Expr>,
    ) -> NodeId {
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

    /// Adds a new input node to the graph.
    ///
    /// # Arguments
    ///
    /// * `dtype` - The data type of the input tensor.
    /// * `shape` - The symbolic shape of the input tensor.
    pub fn input(&self, dtype: DType, shape: Vec<Expr>) -> NodeView {
        let id = self.add_node(TensorOp::Input, vec![], dtype, shape);
        self.inputs.borrow_mut().push(id);
        self.get_view(id)
    }

    /// Gets a `NodeView` for a given `NodeId`.
    pub fn get_view(&self, id: NodeId) -> NodeView {
        NodeView { id, graph: self }
    }

    // --- Internal methods for creating operation nodes ---

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

    pub fn sin(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(TensorOp::Elementwise(AstOp::Sin), vec![src], dtype, shape)
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

    pub fn permute(&self, src: NodeId, axes: Vec<usize>) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = crate::tensor::shape::tracker::ShapeTracker::new(shape);
        let new_shape = tracker.permute(axes.clone()).shape().to_vec();
        self.add_node(TensorOp::Permute(axes), vec![src], dtype, new_shape)
    }

    pub fn contiguous(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(TensorOp::Contiguous, vec![src], dtype, shape)
    }

    pub fn squeeze(&self, src: NodeId, axis: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = crate::tensor::shape::tracker::ShapeTracker::new(shape);
        let new_shape = tracker.squeeze(axis).shape().to_vec();
        self.add_node(TensorOp::Squeeze(axis), vec![src], dtype, new_shape)
    }

    pub fn unsqueeze(&self, src: NodeId, axis: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = crate::tensor::shape::tracker::ShapeTracker::new(shape);
        let new_shape = tracker.unsqueeze(axis).shape().to_vec();
        self.add_node(TensorOp::Unsqueeze(axis), vec![src], dtype, new_shape)
    }

    pub fn expand(&self, src: NodeId, new_shape: Vec<Expr>) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = crate::tensor::shape::tracker::ShapeTracker::new(shape);
        // This just validates the expand operation. The final shape is `new_shape`.
        let _ = tracker.expand(new_shape.clone());
        self.add_node(
            TensorOp::Expand(new_shape.clone()),
            vec![src],
            dtype,
            new_shape,
        )
    }
}

impl<'a> NodeView<'a> {
    /// Returns the operation of the node.
    pub fn op(&self) -> TensorOp {
        self.graph.nodes.borrow()[self.id.0].op.clone()
    }
    /// Returns the source node IDs of the node.
    pub fn src(&self) -> Vec<NodeId> {
        self.graph.nodes.borrow()[self.id.0].src.clone()
    }
    /// Returns the data type of the node.
    pub fn dtype(&self) -> DType {
        self.graph.nodes.borrow()[self.id.0].dtype.clone()
    }
    /// Returns the symbolic shape of the node.
    pub fn shape(&self) -> Vec<Expr> {
        self.graph.nodes.borrow()[self.id.0].shape.clone()
    }

    /// Performs a sum reduction along a specified axis.
    pub fn sum(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.sum(self.id, axis);
        self.graph.get_view(new_id)
    }

    /// Performs a max reduction along a specified axis.
    pub fn max(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.max(self.id, axis);
        self.graph.get_view(new_id)
    }

    /// Performs a product reduction along a specified axis.
    pub fn prod(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.prod(self.id, axis);
        self.graph.get_view(new_id)
    }

    /// Permutes the axes of the tensor.
    pub fn permute(&self, axes: Vec<usize>) -> NodeView<'a> {
        let new_id = self.graph.permute(self.id, axes);
        self.graph.get_view(new_id)
    }

    /// Removes a dimension of size 1 at a specified axis.
    pub fn squeeze(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.squeeze(self.id, axis);
        self.graph.get_view(new_id)
    }

    /// Adds a dimension of size 1 at a specified axis.
    pub fn unsqueeze(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.unsqueeze(self.id, axis);
        self.graph.get_view(new_id)
    }

    /// Expands the tensor to a new shape.
    pub fn expand(&self, new_shape: Vec<Expr>) -> NodeView<'a> {
        let new_id = self.graph.expand(self.id, new_shape);
        self.graph.get_view(new_id)
    }

    /// Returns a contiguous version of the tensor.
    ///
    /// If the tensor is already contiguous, this is a no-op. Otherwise, it
    /// creates a new node that copies the data into a contiguous layout.
    pub fn contiguous(&self) -> NodeView<'a> {
        let new_id = self.graph.contiguous(self.id);
        self.graph.get_view(new_id)
    }

    /// Marks this node as an output of the graph.
    pub fn as_output(&self) -> Self {
        self.graph.outputs.borrow_mut().push(self.id);
        *self
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

impl<'a> NodeView<'a> {
    /// Applies the element-wise sine function.
    pub fn sin(self) -> NodeView<'a> {
        let new_id = self.graph.sin(self.id);
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

        assert_eq!(b.op(), TensorOp::Permute(vec![1, 0]));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), vec![20.into(), 10.into()]);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 1.into(), 20.into()]);
        let b = a.squeeze(1);
        let c = b.unsqueeze(0);

        assert_eq!(b.op(), TensorOp::Squeeze(1));
        assert_eq!(b.shape(), vec![10.into(), 20.into()]);

        assert_eq!(c.op(), TensorOp::Unsqueeze(0));
        assert_eq!(c.shape(), vec![1.into(), 10.into(), 20.into()]);
    }

    #[test]
    fn test_expand() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![1.into(), 20.into()]);
        let new_shape = vec![10.into(), 20.into()];
        let b = a.expand(new_shape.clone());

        assert_eq!(b.op(), TensorOp::Expand(new_shape.clone()));
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

        assert_eq!(b.op(), TensorOp::Contiguous);
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.shape(), a.shape());
    }
}
