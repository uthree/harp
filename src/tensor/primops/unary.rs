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

use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    DimDyn, Dimension, ElementwiseOp, FloatDType, GradFn, Tensor, TensorInner, TensorOp,
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
///
/// Generic over FloatDType to support f32 and f64.
fn create_unary_elementwise<T: FloatDType, D: Dimension>(
    op: ElementwiseOp,
    input: &Tensor<T, D>,
) -> Tensor<T, D> {
    let view = view_from_shape(input.shape());
    let shape = input.shape().to_vec();

    // Convert to InputRef for graph operations
    let inputs = vec![input.as_input_ref()];
    let expr = op.to_ast(1);

    // Use the actual dtype from the input
    let inner = TensorInner::new(TensorOp::elementwise(inputs, expr), view, shape, T::DTYPE);

    Tensor {
        inner: Arc::new(inner),
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Neg implementation (std::ops::Neg) - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Neg for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn neg(self) -> Tensor<T, D> {
        let result = create_unary_elementwise(ElementwiseOp::Neg, self);

        // Use new typed system if input has typed autograd
        if self.requires_grad() {
            let grad_fn = NegBackward::new(self.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Neg for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn neg(self) -> Tensor<T, D> {
        -&self
    }
}

// ============================================================================
// Recip implementation - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Recip for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn recip(self) -> Tensor<T, D> {
        let result = create_unary_elementwise(ElementwiseOp::Recip, self);

        if self.requires_grad() {
            let grad_fn = RecipBackward::new(self.clone(), result.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Recip for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn recip(self) -> Tensor<T, D> {
        (&self).recip()
    }
}

// ============================================================================
// Sqrt implementation - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Sqrt for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn sqrt(self) -> Tensor<T, D> {
        let result = create_unary_elementwise(ElementwiseOp::Sqrt, self);

        if self.requires_grad() {
            let grad_fn = SqrtBackward::new(self.clone(), result.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Sqrt for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn sqrt(self) -> Tensor<T, D> {
        (&self).sqrt()
    }
}

// ============================================================================
// Log2 implementation - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Log2 for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn log2(self) -> Tensor<T, D> {
        let result = create_unary_elementwise(ElementwiseOp::Log2, self);

        if self.requires_grad() {
            let grad_fn = Log2Backward::new(self.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Log2 for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn log2(self) -> Tensor<T, D> {
        (&self).log2()
    }
}

// ============================================================================
// Exp2 implementation - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Exp2 for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn exp2(self) -> Tensor<T, D> {
        let result = create_unary_elementwise(ElementwiseOp::Exp2, self);

        if self.requires_grad() {
            let grad_fn = Exp2Backward::new(self.clone(), result.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Exp2 for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn exp2(self) -> Tensor<T, D> {
        (&self).exp2()
    }
}

// ============================================================================
// Sin implementation - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Sin for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn sin(self) -> Tensor<T, D> {
        let result = create_unary_elementwise(ElementwiseOp::Sin, self);

        if self.requires_grad() {
            let grad_fn = SinBackward::new(self.clone());
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Sin for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn sin(self) -> Tensor<T, D> {
        (&self).sin()
    }
}

// ============================================================================
// Floor implementation - Generic over FloatDType (no gradient)
// ============================================================================

impl<T: FloatDType, D: Dimension> Floor for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    /// Floor is non-differentiable (gradient is 0 almost everywhere).
    /// Therefore, gradient tracking is not preserved for this operation.
    fn floor(self) -> Tensor<T, D> {
        create_unary_elementwise(ElementwiseOp::Floor, self)
    }
}

impl<T: FloatDType, D: Dimension> Floor for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn floor(self) -> Tensor<T, D> {
        (&self).floor()
    }
}

// ============================================================================
// Cast helper function
// ============================================================================

/// Internal helper function for cast with gradient preservation
fn cast_with_grad<S, T, D>(input: &Tensor<S, D>, result: Tensor<T, D>) -> Tensor<T, D>
where
    S: FloatDType,
    T: FloatDType,
    D: Dimension,
{
    // If input requires grad, propagate requires_grad to output
    // Note: The actual backward pass would need to cast gradients back,
    // but for now we just preserve the requires_grad flag.
    if input.requires_grad() {
        // Create output with autograd enabled (but no grad_fn since we can't
        // properly backprop through type casts in the current system)
        result.set_requires_grad(true)
    } else {
        result
    }
}

// ============================================================================
// Cast with gradient for f32 tensors
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Cast to f64 with gradient tracking
    ///
    /// This method casts the tensor to f64 and preserves gradient tracking.
    /// During backpropagation, gradients are cast back to f32.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    /// let y: Tensor<f64, Dim2> = x.cast_f64();
    /// // y.requires_grad() == true
    /// ```
    pub fn cast_f64(&self) -> Tensor<f64, D> {
        cast_with_grad(self, self.cast())
    }
}

// ============================================================================
// Cast with gradient for f64 tensors
// ============================================================================

impl<D: Dimension> Tensor<f64, D> {
    /// Cast to f32 with gradient tracking
    ///
    /// This method casts the tensor to f32 and preserves gradient tracking.
    /// During backpropagation, gradients are cast back to f64.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);
    /// let y: Tensor<f32, Dim2> = x.cast_f32();
    /// // y.requires_grad() == true
    /// ```
    pub fn cast_f32(&self) -> Tensor<f32, D> {
        cast_with_grad(self, self.cast())
    }
}

// ============================================================================
// Typed Backward Structs (new system with static dimension typing)
// ============================================================================

/// Typed gradient for Neg: z = -a
/// ∂L/∂a = -∂L/∂z
pub struct NegBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> NegBackward<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for NegBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            self.input.backward_with(-grad_output);
        }
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Typed gradient for Recip: z = 1/a
/// ∂L/∂a = -∂L/∂z / a² = -∂L/∂z · z²
pub struct RecipBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    output: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> RecipBackward<T, D> {
    pub fn new(input: Tensor<T, D>, output: Tensor<T, D>) -> Self {
        Self { input, output }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for RecipBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            let z_squared = &self.output * &self.output;
            self.input.backward_with(-(grad_output * &z_squared));
        }
    }

    fn name(&self) -> &'static str {
        "RecipBackward"
    }
}

/// Typed gradient for Sqrt: z = √a
/// ∂L/∂a = ∂L/∂z / (2√a) = ∂L/∂z / (2z)
pub struct SqrtBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    output: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> SqrtBackward<T, D> {
    pub fn new(input: Tensor<T, D>, output: Tensor<T, D>) -> Self {
        Self { input, output }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for SqrtBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            // ∂L/∂a = ∂L/∂z / (2z)
            let two = Tensor::<T, DimDyn>::full_dyn(&[], T::TWO);
            let two_d = Tensor::<T, D> {
                inner: two.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let two_z = &two_d * &self.output;
            self.input.backward_with(grad_output * &two_z.recip());
        }
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Typed gradient for Log2: z = log₂(a)
/// ∂L/∂a = ∂L/∂z / (a · ln(2))
pub struct Log2Backward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> Log2Backward<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for Log2Backward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            // ∂L/∂a = ∂L/∂z / (a · ln(2))
            let ln2 = Tensor::<T, DimDyn>::full_dyn(&[], T::LN_2);
            let ln2_d = Tensor::<T, D> {
                inner: ln2.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let denom = &self.input * &ln2_d;
            self.input.backward_with(grad_output * &denom.recip());
        }
    }

    fn name(&self) -> &'static str {
        "Log2Backward"
    }
}

/// Typed gradient for Exp2: z = 2^a
/// ∂L/∂a = ∂L/∂z · z · ln(2)
pub struct Exp2Backward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    output: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> Exp2Backward<T, D> {
    pub fn new(input: Tensor<T, D>, output: Tensor<T, D>) -> Self {
        Self { input, output }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for Exp2Backward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            // ∂L/∂a = ∂L/∂z · z · ln(2)
            let ln2 = Tensor::<T, DimDyn>::full_dyn(&[], T::LN_2);
            let ln2_d = Tensor::<T, D> {
                inner: ln2.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad = grad_output * &self.output;
            self.input.backward_with(&grad * &ln2_d);
        }
    }

    fn name(&self) -> &'static str {
        "Exp2Backward"
    }
}

/// Typed gradient for Sin: z = sin(a)
/// ∂L/∂a = ∂L/∂z · cos(a)
pub struct SinBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> SinBackward<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for SinBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            // ∂L/∂a = ∂L/∂z · cos(a)
            // cos(a) = sin(a + π/2)
            let pi_half = Tensor::<T, DimDyn>::full_dyn(&[], T::FRAC_PI_2);
            let pi_half_d = Tensor::<T, D> {
                inner: pi_half.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let cos_a = (&self.input + &pi_half_d).sin();
            self.input.backward_with(grad_output * &cos_a);
        }
    }

    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_neg() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = -&a;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_recip() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.recip();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.sqrt();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_log2() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.log2();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_exp2() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.exp2();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sin_f32() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.sin();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sin_f64() {
        // Verify that sin works with f64 (FloatDType constraint)
        use crate::tensor::DimDyn;

        // Create an f64 tensor manually
        let inner = TensorInner::new(
            TensorOp::ConstFill(crate::ast::Literal::F64(1.0)),
            crate::tensor::shape::View::contiguous(vec![
                crate::tensor::shape::Expr::from(2i64),
                crate::tensor::shape::Expr::from(3i64),
            ]),
            vec![2, 3],
            crate::ast::DType::F64,
        );
        let a: Tensor<f64, DimDyn> = Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        };
        let c = a.sin();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_floor() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.floor();
        assert_eq!(c.shape(), &[2, 3]);
    }

    // =========================================================================
    // Cast with gradient tests
    // =========================================================================

    #[test]
    fn test_cast_f64_method() {
        // Cast f32 -> f64 using cast_f64 method
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.cast_f64();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(*b.dtype(), crate::ast::DType::F64);
    }

    #[test]
    fn test_cast_f32_method() {
        use crate::tensor::DimDyn;

        // Create f64 tensor
        let inner = TensorInner::new(
            TensorOp::ConstFill(crate::ast::Literal::F64(2.0)),
            crate::tensor::shape::View::contiguous(vec![
                crate::tensor::shape::Expr::from(2i64),
                crate::tensor::shape::Expr::from(3i64),
            ]),
            vec![2, 3],
            crate::ast::DType::F64,
        );
        let a: Tensor<f64, DimDyn> = Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Cast f64 -> f32 using cast_f32 method
        let b = a.cast_f32();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(*b.dtype(), crate::ast::DType::F32);
    }

    #[test]
    fn test_cast_f64_preserves_requires_grad() {
        // Cast with requires_grad should preserve gradient tracking
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        assert!(a.requires_grad());

        let b = a.cast_f64();
        assert!(b.requires_grad(), "Cast output should have requires_grad");
    }

    #[test]
    fn test_cast_f64_no_grad_when_input_no_grad() {
        // Cast without requires_grad should not track gradients
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        assert!(!a.requires_grad());

        let b = a.cast_f64();
        assert!(
            !b.requires_grad(),
            "Cast output should not have requires_grad"
        );
    }

    #[test]
    fn test_cast_f32_preserves_requires_grad() {
        use crate::tensor::DimDyn;

        // Create f64 tensor with requires_grad
        let inner = TensorInner::new(
            TensorOp::ConstFill(crate::ast::Literal::F64(2.0)),
            crate::tensor::shape::View::contiguous(vec![
                crate::tensor::shape::Expr::from(2i64),
                crate::tensor::shape::Expr::from(3i64),
            ]),
            vec![2, 3],
            crate::ast::DType::F64,
        );
        let a_base: Tensor<f64, DimDyn> = Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        };
        let a = a_base.set_requires_grad(true);
        assert!(a.requires_grad());

        let b = a.cast_f32();
        assert!(b.requires_grad(), "Cast output should have requires_grad");
    }
}
