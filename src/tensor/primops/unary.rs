//! Unary primitive operations
//!
//! - Neg: negation (-x)
//! - Recip: reciprocal (1/x)
//! - Sqrt: square root
//! - Log2: base-2 logarithm
//! - Exp2: base-2 exponential
//! - Sin: sine

use std::marker::PhantomData;
use std::ops::Neg;
use std::sync::Arc;

use crate::ast::DType;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{Dimension, ElementwiseOp, Tensor, TensorInner, TensorOp};

use super::binary::with_grad_fn;
use super::grad::{
    Exp2Backward, Log2Backward, NegBackward, RecipBackward, SinBackward, SqrtBackward,
};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

/// Create a unary elementwise Tensor using Compute variant
fn create_unary_elementwise<D: Dimension>(op: ElementwiseOp, input: &Tensor<D>) -> Tensor<D> {
    let view = view_from_shape(input.shape());
    let shape = input.shape().to_vec();

    // Create Compute operation with input embedded
    let inputs = vec![Arc::new(input.clone().into_dyn())];
    let expr = op.to_ast(1);

    let inner = TensorInner::new(TensorOp::elementwise(inputs, expr), view, shape, DType::F32);

    Tensor {
        inner: Arc::new(inner),
        _dim: PhantomData,
    }
}

// ============================================================================
// Neg: -Tensor
// ============================================================================

impl<D: Dimension> Neg for &Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Neg, self);

        if self.requires_grad() {
            let grad_fn = NegBackward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Neg for Tensor<D> {
    type Output = Tensor<D>;
    fn neg(self) -> Tensor<D> {
        -&self
    }
}

// ============================================================================
// Unary primops as methods
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Compute the reciprocal (1/x) of each element (primop)
    pub fn recip(&self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Recip, self);

        if self.requires_grad() {
            let grad_fn = RecipBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute sqrt(x) for each element (primop)
    pub fn sqrt(&self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Sqrt, self);

        if self.requires_grad() {
            let grad_fn = SqrtBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute log2(x) for each element (primop)
    pub fn log2(&self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Log2, self);

        if self.requires_grad() {
            let grad_fn = Log2Backward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute exp2(x) = 2^x for each element (primop)
    pub fn exp2(&self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Exp2, self);

        if self.requires_grad() {
            let grad_fn = Exp2Backward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute sin(x) for each element (primop)
    pub fn sin(&self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Sin, self);

        if self.requires_grad() {
            let grad_fn = SinBackward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute floor(x) for each element (primop)
    ///
    /// Floor is non-differentiable (gradient is 0 almost everywhere).
    /// Therefore, gradient tracking is not preserved for this operation.
    pub fn floor(&self) -> Tensor<D> {
        // floor is non-differentiable, so we don't track gradients
        create_unary_elementwise(ElementwiseOp::Floor, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_neg() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = -&a;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_recip() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.recip();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.sqrt();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_log2() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.log2();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_exp2() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.exp2();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sin() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.sin();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_floor() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.floor();
        assert_eq!(c.shape(), &[2, 3]);
    }
}
