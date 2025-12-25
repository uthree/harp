//! Initialization operations (primops)
//!
//! - Const: constant tensor
//! - Rand: uniform random [0, 1)

use std::marker::PhantomData;

use crate::ast::Literal;
use crate::core::DType;
use crate::core::shape::{Expr, View};
use crate::tensor::{Dim, DimDyn, Dimension, Tensor, TensorNode, TensorOp};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

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
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(
            TensorOp::ConstFill(Literal::F32(value)),
            vec![],
            view,
            DType::F32,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create an input tensor (placeholder for data)
    pub fn input(name: &str, shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
            DType::F32,
            name,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create a tensor with uniform random values [0, 1)
    pub fn rand(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(TensorOp::Rand, vec![], view, DType::F32);
        Self::from_tensor_node(tensor_node, shape_vec)
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
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(
            TensorOp::ConstFill(Literal::F32(value)),
            vec![],
            view,
            DType::F32,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create an input tensor (dynamic shape)
    pub fn input_dyn(name: &str, shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
            DType::F32,
            name,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create a tensor with uniform random values [0, 1) (dynamic shape)
    pub fn rand_dyn(shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(TensorOp::Rand, vec![], view, DType::F32);
        Self::from_tensor_node(tensor_node, shape_vec)
    }
}

// ============================================================================
// Internal constructor
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Create a new tensor from a TensorNode directly
    pub(crate) fn from_tensor_node(tensor_node: TensorNode, shape: Vec<usize>) -> Self {
        use std::cell::RefCell;
        use std::rc::Rc;

        let dtype = tensor_node.dtype.clone();
        Self {
            inner: Rc::new(tensor_node),
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
