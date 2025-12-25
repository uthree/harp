//! Unary primitive operations
//!
//! Each operation is defined as a separate trait for flexibility.
//!
//! - Neg: negation (-x) - uses std::ops::Neg
//! - Recip: reciprocal (1/x)
//! - Sqrt: square root
//! - Log2: base-2 logarithm
//! - Exp2: base-2 exponential
//! - Sin: sine
//! - Floor: floor function

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

// ============================================================================
// Unary operation traits
// ============================================================================

/// Reciprocal operation (1/x)
pub trait Recip {
    type Output;
    fn recip(self) -> Self::Output;
}

/// Square root operation
pub trait Sqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}

/// Base-2 logarithm operation
pub trait Log2 {
    type Output;
    fn log2(self) -> Self::Output;
}

/// Base-2 exponential operation (2^x)
pub trait Exp2 {
    type Output;
    fn exp2(self) -> Self::Output;
}

/// Sine operation
pub trait Sin {
    type Output;
    fn sin(self) -> Self::Output;
}

/// Floor operation
pub trait Floor {
    type Output;
    fn floor(self) -> Self::Output;
}

// ============================================================================
// Helper functions
// ============================================================================

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
// Neg implementation (std::ops::Neg)
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
// Recip implementation
// ============================================================================

impl<D: Dimension> Recip for &Tensor<D> {
    type Output = Tensor<D>;

    fn recip(self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Recip, self);

        if self.requires_grad() {
            let grad_fn = RecipBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Recip for Tensor<D> {
    type Output = Tensor<D>;
    fn recip(self) -> Tensor<D> {
        (&self).recip()
    }
}

// ============================================================================
// Sqrt implementation
// ============================================================================

impl<D: Dimension> Sqrt for &Tensor<D> {
    type Output = Tensor<D>;

    fn sqrt(self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Sqrt, self);

        if self.requires_grad() {
            let grad_fn = SqrtBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Sqrt for Tensor<D> {
    type Output = Tensor<D>;
    fn sqrt(self) -> Tensor<D> {
        (&self).sqrt()
    }
}

// ============================================================================
// Log2 implementation
// ============================================================================

impl<D: Dimension> Log2 for &Tensor<D> {
    type Output = Tensor<D>;

    fn log2(self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Log2, self);

        if self.requires_grad() {
            let grad_fn = Log2Backward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Log2 for Tensor<D> {
    type Output = Tensor<D>;
    fn log2(self) -> Tensor<D> {
        (&self).log2()
    }
}

// ============================================================================
// Exp2 implementation
// ============================================================================

impl<D: Dimension> Exp2 for &Tensor<D> {
    type Output = Tensor<D>;

    fn exp2(self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Exp2, self);

        if self.requires_grad() {
            let grad_fn = Exp2Backward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Exp2 for Tensor<D> {
    type Output = Tensor<D>;
    fn exp2(self) -> Tensor<D> {
        (&self).exp2()
    }
}

// ============================================================================
// Sin implementation
// ============================================================================

impl<D: Dimension> Sin for &Tensor<D> {
    type Output = Tensor<D>;

    fn sin(self) -> Tensor<D> {
        let result = create_unary_elementwise(ElementwiseOp::Sin, self);

        if self.requires_grad() {
            let grad_fn = SinBackward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Sin for Tensor<D> {
    type Output = Tensor<D>;
    fn sin(self) -> Tensor<D> {
        (&self).sin()
    }
}

// ============================================================================
// Floor implementation
// ============================================================================

impl<D: Dimension> Floor for &Tensor<D> {
    type Output = Tensor<D>;

    /// Floor is non-differentiable (gradient is 0 almost everywhere).
    /// Therefore, gradient tracking is not preserved for this operation.
    fn floor(self) -> Tensor<D> {
        create_unary_elementwise(ElementwiseOp::Floor, self)
    }
}

impl<D: Dimension> Floor for Tensor<D> {
    type Output = Tensor<D>;
    fn floor(self) -> Tensor<D> {
        (&self).floor()
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
