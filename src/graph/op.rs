use crate::{
    ast::{AstNode, AstOp, Const},
    graph::shape::expr::Expr,
};

/// An enumeration of all possible tensor operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphOp {
    /// An input tensor to the graph.
    Input,
    /// A tensor filled with a single constant value.
    Full(Const),
    /// A tensor filled with random values.
    Rand,
    /// An element-wise operation (e.g., add, mul, sin).
    Elementwise(AstOp),
    /// A reduction operation along a specific axis (e.g., sum, max).
    Reduce(AstOp, usize),
    /// A cumulative operation along a specific axis (e.g., cumsum).
    Cumulative(AstOp, usize),
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
    /// An operation that slices a tensor.
    Slice(Vec<(Expr, Expr)>),
    /// An operation that creates a sliding window view of a tensor.
    Unfold1d {
        dim: usize,
        kernel_size: usize,
        stride: usize,
    },
    Unfold2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
    /// An operation that changes the shape of a tensor without changing its data.
    Reshape(Vec<Expr>),

    // Fused Operators
    FusedElementwise(AstNode), // Capture(n) means n-th src element
    FusedElementwiseReduce(AstNode, AstOp, Vec<usize>), // Capture(n) means n-th src element, reduce some axis
    FusedReduce(AstOp, Vec<usize>),                     // reduce multiple axis
}

impl GraphOp {
    /// Returns `true` if the operation is an element-wise operation that can be fused.
    pub fn is_elementwise(&self) -> bool {
        matches!(self, GraphOp::Elementwise(_))
    }

    /// Returns `true` if the operation is a `Full` (constant) operation.
    pub fn is_full(&self) -> bool {
        matches!(self, GraphOp::Full(_))
    }
}
