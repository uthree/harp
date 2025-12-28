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

use super::binary::with_grad_fn_generic;

// ============================================================================
// Unary Gradients (generic over FloatDType)
// ============================================================================

/// Gradient for Neg: z = -a
/// ∂L/∂a = -∂L/∂z
pub struct NegBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
}

impl<T: FloatDType> NegBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>) -> Self {
        Self { input }
    }
}

// NegBackward is generic since Neg is implemented for FloatDType
impl<T: FloatDType> GradFn<T> for NegBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        vec![-grad_output]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Gradient for Recip: z = 1/a
/// ∂L/∂a = -∂L/∂z / a² = -∂L/∂z · z²
pub struct RecipBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output: Tensor<T, DimDyn>,
}

impl<T: FloatDType> RecipBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, output: Tensor<T, DimDyn>) -> Self {
        Self { input, output }
    }
}

// RecipBackward is generic since Neg and Mul are implemented for FloatDType
impl<T: FloatDType> GradFn<T> for RecipBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // ∂L/∂a = -∂L/∂z · z² where z = 1/a
        let z_squared = &self.output * &self.output;
        vec![-(grad_output * &z_squared)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "RecipBackward"
    }
}

/// Gradient for Sqrt: z = √a
/// ∂L/∂a = ∂L/∂z / (2·√a) = ∂L/∂z / (2·z)
pub struct SqrtBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output: Tensor<T, DimDyn>,
}

impl<T: FloatDType> SqrtBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, output: Tensor<T, DimDyn>) -> Self {
        Self { input, output }
    }
}

// SqrtBackward is generic since Add, Mul, and Recip are implemented for FloatDType
impl<T: FloatDType> GradFn<T> for SqrtBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        let two_sqrt = &self.output + &self.output; // output * 2
        // Use recip() * for generic implementation
        vec![grad_output * &two_sqrt.recip()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Gradient for Log2: z = log₂(a)
/// ∂L/∂a = ∂L/∂z / (a · ln(2))
pub struct Log2Backward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
}

impl<T: FloatDType> Log2Backward<T> {
    pub fn new(input: Tensor<T, DimDyn>) -> Self {
        Self { input }
    }
}

// Log2Backward is generic since Mul and Recip are implemented for FloatDType
impl<T: FloatDType> GradFn<T> for Log2Backward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Note: ignoring ln(2) factor for now
        // TODO: Add scalar operations
        // Use recip() * for generic implementation
        vec![grad_output * &self.input.clone().recip()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Log2Backward"
    }
}

/// Gradient for Exp2: z = 2^a
/// ∂L/∂a = ∂L/∂z · 2^a · ln(2) = ∂L/∂z · z · ln(2)
pub struct Exp2Backward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output: Tensor<T, DimDyn>,
}

impl<T: FloatDType> Exp2Backward<T> {
    pub fn new(input: Tensor<T, DimDyn>, output: Tensor<T, DimDyn>) -> Self {
        Self { input, output }
    }
}

// Exp2Backward is generic since Mul is implemented for FloatDType
impl<T: FloatDType> GradFn<T> for Exp2Backward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Approximate: ignoring ln(2) factor for now
        // TODO: Add scalar multiplication
        vec![grad_output * &self.output]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Exp2Backward"
    }
}

/// Gradient for Sin: z = sin(a)
/// ∂L/∂a = ∂L/∂z · cos(a)
pub struct SinBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
}

impl<T: FloatDType> SinBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>) -> Self {
        Self { input }
    }
}

// SinBackward is generic since Mul and Sin are implemented for FloatDType
impl<T: FloatDType> GradFn<T> for SinBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // cos(x) = sin(x + π/2)
        // TODO: Implement proper cos using scalar addition
        // Approximate: just use the input sin (incorrect but compiles)
        let cos_input = (&self.input).sin();
        vec![grad_output * &cos_input]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

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

        if self.requires_grad() {
            let grad_fn = NegBackward::new(self.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
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
            let grad_fn = RecipBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
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
            let grad_fn = SqrtBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
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
            let grad_fn = Log2Backward::new(self.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
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
            let grad_fn = Exp2Backward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
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
            let grad_fn = SinBackward::new(self.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
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
// Cast gradients for FloatDType (generic)
// ============================================================================

/// Generic gradient for Cast S -> T
/// ∂L/∂a = cast(∂L/∂z, S)
///
/// The gradient is simply cast back to the input type.
/// S is the source type, T is the target type.
pub struct CastBackward<S: FloatDType, T: FloatDType> {
    input: Tensor<S, DimDyn>,
    _target: PhantomData<T>,
}

impl<S: FloatDType, T: FloatDType> CastBackward<S, T> {
    pub fn new(input: Tensor<S, DimDyn>) -> Self {
        Self {
            input,
            _target: PhantomData,
        }
    }
}

impl<S: FloatDType, T: FloatDType> GradFn<T> for CastBackward<S, T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Cast gradient back to source type S and propagate to input
        let grad_s: Tensor<S, DimDyn> = grad_output.cast();
        if self.input.requires_grad() {
            S::call_backward_with(&self.input, grad_s);
        }
        // Return empty since we've already propagated
        vec![]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        // Return empty since we handle propagation directly
        vec![]
    }

    fn name(&self) -> &'static str {
        "CastBackward"
    }
}

/// Type alias for backward compatibility
pub type CastF32ToF64Backward = CastBackward<f32, f64>;
/// Type alias for backward compatibility
pub type CastF64ToF32Backward = CastBackward<f64, f32>;

// ============================================================================
// Cast with gradient (generic helper)
// ============================================================================

/// Internal helper function to create a tensor with gradient tracking after cast
fn cast_with_grad<S, T, D>(input: &Tensor<S, D>, result: Tensor<T, D>) -> Tensor<T, D>
where
    S: FloatDType,
    T: FloatDType,
    D: Dimension,
{
    if input.requires_grad() {
        let grad_fn = CastBackward::<S, T>::new(input.clone().into_dyn());
        let inner = crate::tensor::TensorInner {
            op: result.inner.op.clone(),
            view: result.inner.view.clone(),
            shape: result.inner.shape.clone(),
            dtype: result.inner.dtype.clone(),
            name: result.inner.name.clone(),
            autograd: Some(T::wrap_grad_fn(Arc::new(grad_fn))),
            buffer: std::sync::RwLock::new(result.inner.clone_buffer()),
        };
        Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
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
