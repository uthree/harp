//! Initialization operations (primops)
//!
//! - Const: constant tensor
//! - Rand: uniform random [0, 1)

use std::marker::PhantomData;

use crate::graph::{DType, GraphNode, GraphOp, shape::View};
use crate::tensor::{Dim, DimDyn, Dimension, Tensor};

// ============================================================================
// Static dimension constructors
// ============================================================================

impl<const N: usize> Tensor<Dim<N>>
where
    Dim<N>: Dimension,
{
    /// Create a tensor filled with zeros (Const(0))
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::full(shape, 0.0)
    }

    /// Create a tensor filled with ones (Const(1))
    pub fn ones(shape: [usize; N]) -> Self {
        Self::full(shape, 1.0)
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: [usize; N], value: f32) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let node = GraphNode::full(value, shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create an input tensor (placeholder for data)
    pub fn input(name: &str, shape: [usize; N]) -> Self {
        use crate::graph::shape::Expr;
        let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::contiguous(shape_exprs);
        let node = GraphNode::new(
            DType::F32,
            GraphOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
        );
        Self::from_node(node, shape.to_vec(), DType::F32)
    }

    /// Create a tensor with uniform random values [0, 1)
    pub fn rand(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let node = GraphNode::rand(shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }
}

// ============================================================================
// Dynamic dimension constructors
// ============================================================================

impl Tensor<DimDyn> {
    /// Create a tensor filled with zeros (dynamic shape)
    pub fn zeros_dyn(shape: &[usize]) -> Self {
        Self::full_dyn(shape, 0.0)
    }

    /// Create a tensor filled with ones (dynamic shape)
    pub fn ones_dyn(shape: &[usize]) -> Self {
        Self::full_dyn(shape, 1.0)
    }

    /// Create a tensor filled with a constant value (dynamic shape)
    pub fn full_dyn(shape: &[usize], value: f32) -> Self {
        let shape_vec = shape.to_vec();
        let node = GraphNode::full(value, shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create an input tensor (dynamic shape)
    pub fn input_dyn(name: &str, shape: &[usize]) -> Self {
        use crate::graph::shape::Expr;
        let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
        let view = View::contiguous(shape_exprs);
        let node = GraphNode::new(
            DType::F32,
            GraphOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
        );
        Self::from_node(node, shape.to_vec(), DType::F32)
    }

    /// Create a tensor with uniform random values [0, 1) (dynamic shape)
    pub fn rand_dyn(shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let node = GraphNode::rand(shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }
}

// ============================================================================
// Internal constructor
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Create a new tensor from a graph node (internal use)
    pub(crate) fn from_node(node: GraphNode, shape: Vec<usize>, dtype: DType) -> Self {
        use std::cell::RefCell;
        Self {
            node,
            shape,
            dtype,
            autograd: None,
            buffer: RefCell::new(None),
            _dim: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim1, Dim2, Dim3};

    #[test]
    fn test_zeros() {
        let t = Tensor::<Dim2>::zeros([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_ones() {
        let t = Tensor::<Dim3>::ones([2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_full() {
        let t = Tensor::<Dim1>::full([10], 2.5);
        assert_eq!(t.shape(), &[10]);
    }

    #[test]
    fn test_rand() {
        let t = Tensor::<Dim2>::rand([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_zeros_dyn() {
        let t = Tensor::<DimDyn>::zeros_dyn(&[3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
    }

    #[test]
    fn test_rand_dyn() {
        let t = Tensor::<DimDyn>::rand_dyn(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }
}
