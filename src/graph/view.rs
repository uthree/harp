use super::{Graph, GraphOp, NodeId};
use crate::ast::DType;
use crate::graph::shape::expr::Expr;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// A temporary, lightweight handle to a node in the graph.
///
/// `NodeView` provides a convenient, chainable API for building the computation graph.
/// It holds a reference to the graph and the ID of the node it represents.
/// Most tensor operations are implemented on `NodeView`.
///
/// # Examples
///
/// ```
/// use harp::graph::Graph;
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

impl<'a> NodeView<'a> {
    /// Returns the operation of the node.
    pub fn op(&self) -> GraphOp {
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

    /// Performs a cumulative sum along a specified axis.
    pub fn cumsum(&self, axis: usize) -> NodeView<'a> {
        let new_id = self.graph.cumsum(self.id, axis);
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

    /// Slices the tensor.
    pub fn slice(&self, args: Vec<(Expr, Expr)>) -> NodeView<'a> {
        let new_id = self.graph.slice(self.id, args);
        self.graph.get_view(new_id)
    }

    /// Creates a sliding window view of the tensor.
    pub fn unfold1d(&self, dim: usize, kernel_size: usize, stride: usize) -> NodeView<'a> {
        let new_id = self.graph.unfold1d(self.id, dim, kernel_size, stride);
        self.graph.get_view(new_id)
    }

    /// Creates a 2D sliding window view of the tensor.
    pub fn unfold2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> NodeView<'a> {
        let new_id = self.graph.unfold2d(self.id, kernel_size, stride);
        self.graph.get_view(new_id)
    }

    /// Reshapes the tensor to a new shape.
    pub fn reshape(&self, new_shape: Vec<Expr>) -> NodeView<'a> {
        let new_id = self.graph.reshape(self.id, new_shape);
        self.graph.get_view(new_id)
    }

    /// Performs a 1D convolution.
    pub fn conv1d(
        self,
        weight: NodeView<'a>,
        kernel_size: usize,
        stride: usize,
        groups: usize,
    ) -> NodeView<'a> {
        let new_id = self
            .graph
            .conv1d(self.id, weight.id, kernel_size, stride, groups);
        self.graph.get_view(new_id)
    }

    /// Performs a 2D convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        self,
        weight: NodeView<'a>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> NodeView<'a> {
        let new_id = self
            .graph
            .conv2d(self.id, weight.id, kernel_size, stride, groups);
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

    /// Performs an element-wise less-than comparison.
    pub fn lt(self, rhs: Self) -> NodeView<'a> {
        let new_id = self.graph.lt(self.id, rhs.id);
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

impl<'a> Rem for NodeView<'a> {
    type Output = NodeView<'a>;
    fn rem(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.rem(self.id, rhs.id);
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

    /// Applies the element-wise square root function.
    pub fn sqrt(self) -> NodeView<'a> {
        let new_id = self.graph.sqrt(self.id);
        self.graph.get_view(new_id)
    }

    /// Applies the element-wise base-2 logarithm function.
    pub fn log2(self) -> NodeView<'a> {
        let new_id = self.graph.log2(self.id);
        self.graph.get_view(new_id)
    }

    /// Applies the element-wise base-2 exponential function.
    pub fn exp2(self) -> NodeView<'a> {
        let new_id = self.graph.exp2(self.id);
        self.graph.get_view(new_id)
    }

    /// Applies the element-wise reciprocal (1/x) function.
    pub fn recip(self) -> NodeView<'a> {
        let new_id = self.graph.recip(self.id);
        self.graph.get_view(new_id)
    }
}
