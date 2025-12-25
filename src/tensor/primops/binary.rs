//! Binary primitive operations
//!
//! - Add: element-wise addition
//! - Mul: element-wise multiplication
//! - Max: element-wise maximum
//! - Idiv: integer division

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Add;
use std::ops::Mul;
use std::rc::Rc;

use crate::ast::Literal;
use crate::core::DType;
use crate::core::shape::{Expr, View};
use crate::tensor::{
    DimDyn, Dimension, ElementwiseOp, GradFn, Tensor, TensorData, TensorNode, TensorOp,
};

use super::grad::{AddBackward, MaxBackward, MulBackward};

// ============================================================================
// Helper functions
// ============================================================================

/// Check if any input requires gradients
pub(crate) fn any_requires_grad<D1: Dimension, D2: Dimension>(
    a: &Tensor<D1>,
    b: &Tensor<D2>,
) -> bool {
    a.requires_grad() || b.requires_grad()
}

/// Create a tensor with gradient tracking if needed
pub(crate) fn with_grad_fn<D: Dimension>(
    tensor: Tensor<D>,
    grad_fn: Option<Rc<dyn GradFn>>,
) -> Tensor<D> {
    if grad_fn.is_some() {
        Tensor {
            inner: tensor.inner,
            shape: tensor.shape,
            dtype: tensor.dtype,
            autograd: Some(Rc::new(TensorData {
                requires_grad: true,
                grad: RefCell::new(None),
                grad_fn,
                cached_data: RefCell::new(None),
            })),
            buffer: tensor.buffer,
            _dim: PhantomData,
        }
    } else {
        tensor
    }
}

/// Helper to compute result shape for binary operations with broadcasting
pub(crate) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_dim = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let b_dim = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            panic!(
                "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                a, b, i
            );
        }
    }
    result
}

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

/// Create a binary elementwise TensorNode
fn create_binary_elementwise<D: Dimension>(
    op: ElementwiseOp,
    lhs: &Tensor<D>,
    rhs: &Tensor<impl Dimension>,
) -> TensorNode {
    let result_shape = broadcast_shapes(lhs.shape(), rhs.shape());
    let view = view_from_shape(&result_shape);
    TensorNode::new(
        TensorOp::Elementwise { op },
        vec![lhs.clone().into_dyn(), rhs.clone().into_dyn()],
        view,
        DType::F32,
    )
}

/// Create a tensor with scalar (constant fill) for binary ops
fn scalar_tensor(value: f32, _shape: &[usize]) -> Tensor<DimDyn> {
    // For scalar operations, we create a scalar (shape=[]) tensor
    // that will be broadcast to the target shape
    let view = View::contiguous(Vec::<Expr>::new()); // scalar
    let tensor_node = TensorNode::new(
        TensorOp::ConstFill(Literal::F32(value)),
        vec![],
        view,
        DType::F32,
    );
    Tensor::from_tensor_node(tensor_node, vec![])
}

// ============================================================================
// Add: Tensor + Tensor
// ============================================================================

impl<D: Dimension> Add for &Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Self) -> Tensor<D> {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let tensor_node = create_binary_elementwise(ElementwiseOp::Add, self, rhs);
        let result = Tensor::from_tensor_node(tensor_node, result_shape);

        if any_requires_grad(self, rhs) {
            let grad_fn = AddBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Add<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;
    fn add(self, rhs: Tensor<D>) -> Tensor<D> {
        self + &rhs
    }
}

impl<D: Dimension> Add<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;
    fn add(self, rhs: &Tensor<D>) -> Tensor<D> {
        &self + rhs
    }
}

impl<D: Dimension> Add for Tensor<D> {
    type Output = Tensor<D>;
    fn add(self, rhs: Self) -> Tensor<D> {
        &self + &rhs
    }
}

// Add: Tensor + f32
impl<D: Dimension> Add<f32> for &Tensor<D> {
    type Output = Tensor<D>;
    fn add(self, rhs: f32) -> Tensor<D> {
        let scalar = scalar_tensor(rhs, &[]);
        let tensor_node = create_binary_elementwise(ElementwiseOp::Add, self, &scalar);
        Tensor::from_tensor_node(tensor_node, self.shape().to_vec())
    }
}

impl<D: Dimension> Add<f32> for Tensor<D> {
    type Output = Tensor<D>;
    fn add(self, rhs: f32) -> Tensor<D> {
        &self + rhs
    }
}

// Add: f32 + Tensor
impl<D: Dimension> Add<&Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn add(self, rhs: &Tensor<D>) -> Tensor<D> {
        let scalar = scalar_tensor(self, &[]);
        let tensor_node = create_binary_elementwise(ElementwiseOp::Add, &scalar, rhs);
        Tensor::from_tensor_node(tensor_node, rhs.shape().to_vec())
    }
}

impl<D: Dimension> Add<Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn add(self, rhs: Tensor<D>) -> Tensor<D> {
        self + &rhs
    }
}

// ============================================================================
// Mul: Tensor * Tensor
// ============================================================================

impl<D: Dimension> Mul for &Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Self) -> Tensor<D> {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let tensor_node = create_binary_elementwise(ElementwiseOp::Mul, self, rhs);
        let result = Tensor::from_tensor_node(tensor_node, result_shape);

        if any_requires_grad(self, rhs) {
            let grad_fn = MulBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Mul<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;
    fn mul(self, rhs: Tensor<D>) -> Tensor<D> {
        self * &rhs
    }
}

impl<D: Dimension> Mul<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;
    fn mul(self, rhs: &Tensor<D>) -> Tensor<D> {
        &self * rhs
    }
}

impl<D: Dimension> Mul for Tensor<D> {
    type Output = Tensor<D>;
    fn mul(self, rhs: Self) -> Tensor<D> {
        &self * &rhs
    }
}

// Mul: Tensor * f32
impl<D: Dimension> Mul<f32> for &Tensor<D> {
    type Output = Tensor<D>;
    fn mul(self, rhs: f32) -> Tensor<D> {
        let scalar = scalar_tensor(rhs, &[]);
        let tensor_node = create_binary_elementwise(ElementwiseOp::Mul, self, &scalar);
        Tensor::from_tensor_node(tensor_node, self.shape().to_vec())
    }
}

impl<D: Dimension> Mul<f32> for Tensor<D> {
    type Output = Tensor<D>;
    fn mul(self, rhs: f32) -> Tensor<D> {
        &self * rhs
    }
}

// Mul: f32 * Tensor
impl<D: Dimension> Mul<&Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn mul(self, rhs: &Tensor<D>) -> Tensor<D> {
        let scalar = scalar_tensor(self, &[]);
        let tensor_node = create_binary_elementwise(ElementwiseOp::Mul, &scalar, rhs);
        Tensor::from_tensor_node(tensor_node, rhs.shape().to_vec())
    }
}

impl<D: Dimension> Mul<Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn mul(self, rhs: Tensor<D>) -> Tensor<D> {
        self * &rhs
    }
}

// ============================================================================
// Max: element-wise maximum
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Compute element-wise maximum with another tensor (primop)
    pub fn max(&self, other: &Tensor<impl Dimension>) -> Tensor<D> {
        let result_shape = broadcast_shapes(self.shape(), other.shape());
        let tensor_node = create_binary_elementwise(ElementwiseOp::Max, self, other);
        let result = Tensor::from_tensor_node(tensor_node, result_shape);

        if self.requires_grad() || other.requires_grad() {
            let grad_fn = MaxBackward::new(self.clone().into_dyn(), other.clone().into_dyn());
            with_grad_fn(result, Some(Rc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute element-wise maximum with a scalar
    pub fn max_scalar(&self, value: f32) -> Tensor<D> {
        let scalar = scalar_tensor(value, &[]);
        self.max(&scalar)
    }
}

// ============================================================================
// Idiv: integer division (floor division)
// TODO: Requires floor() support in graph ops
// ============================================================================

// impl<D: Dimension> Tensor<D> {
//     /// Integer division (floor division)
//     pub fn idiv(&self, other: &Tensor<impl Dimension>) -> Tensor<D> {
//         // Requires floor() primop support
//         todo!("idiv requires floor() support in graph ops")
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_add_tensors() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = Tensor::<Dim2>::ones([2, 3]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::<Dim2>::zeros([2, 3]);
        let c = &a + 1.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_mul_tensors() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = Tensor::<Dim2>::full([2, 3], 2.0);
        let c = &a * &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_max() {
        let a = Tensor::<Dim2>::full([2, 3], -1.0);
        let b = Tensor::<Dim2>::full([2, 3], 0.0);
        let c = a.max(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_max_scalar() {
        let a = Tensor::<Dim2>::full([2, 3], -1.0);
        let c = a.max_scalar(0.0);
        assert_eq!(c.shape(), &[2, 3]);
    }
}
