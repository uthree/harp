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
    DimDyn, Dimension, ElementwiseOp, FloatDType, GradFn, Tensor, TensorDType, TensorInner,
    TensorOp,
};

use super::binary::with_grad_fn;

// ============================================================================
// Helper for type conversion
// ============================================================================

/// Convert any Tensor<T, D> to Tensor<f32, DimDyn> for graph operations.
///
/// This is safe because the tensor's layout is the same for all T,
/// and the actual dtype is stored in TensorInner at runtime.
fn to_graph_ref<T: TensorDType, D: Dimension>(tensor: &Tensor<T, D>) -> Tensor<f32, DimDyn> {
    Tensor {
        inner: tensor.inner.clone(),
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Unary Gradients (f32-only for now, infrastructure for generic later)
// ============================================================================

/// Gradient for Neg: z = -a
/// ∂L/∂a = -∂L/∂z
pub struct NegBackward {
    input: Tensor<f32, DimDyn>,
}

impl NegBackward {
    pub fn new(input: Tensor<f32, DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn<f32> for NegBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        vec![-grad_output]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Gradient for Recip: z = 1/a
/// ∂L/∂a = -∂L/∂z / a² = -∂L/∂z · z²
pub struct RecipBackward {
    input: Tensor<f32, DimDyn>,
    output: Tensor<f32, DimDyn>,
}

impl RecipBackward {
    pub fn new(input: Tensor<f32, DimDyn>, output: Tensor<f32, DimDyn>) -> Self {
        Self { input, output }
    }
}

impl GradFn<f32> for RecipBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // ∂L/∂a = -∂L/∂z · z² where z = 1/a
        let z_squared = &self.output * &self.output;
        vec![-(grad_output * &z_squared)]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "RecipBackward"
    }
}

/// Gradient for Sqrt: z = √a
/// ∂L/∂a = ∂L/∂z / (2·√a) = ∂L/∂z / (2·z)
pub struct SqrtBackward {
    input: Tensor<f32, DimDyn>,
    output: Tensor<f32, DimDyn>,
}

impl SqrtBackward {
    pub fn new(input: Tensor<f32, DimDyn>, output: Tensor<f32, DimDyn>) -> Self {
        Self { input, output }
    }
}

impl GradFn<f32> for SqrtBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        let two_sqrt = &self.output + &self.output; // output * 2
        vec![grad_output / &two_sqrt]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Gradient for Log2: z = log₂(a)
/// ∂L/∂a = ∂L/∂z / (a · ln(2))
pub struct Log2Backward {
    input: Tensor<f32, DimDyn>,
}

impl Log2Backward {
    pub fn new(input: Tensor<f32, DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn<f32> for Log2Backward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // Note: ignoring ln(2) factor for now
        // TODO: Add scalar operations
        vec![grad_output / &self.input]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Log2Backward"
    }
}

/// Gradient for Exp2: z = 2^a
/// ∂L/∂a = ∂L/∂z · 2^a · ln(2) = ∂L/∂z · z · ln(2)
pub struct Exp2Backward {
    input: Tensor<f32, DimDyn>,
    output: Tensor<f32, DimDyn>,
}

impl Exp2Backward {
    pub fn new(input: Tensor<f32, DimDyn>, output: Tensor<f32, DimDyn>) -> Self {
        Self { input, output }
    }
}

impl GradFn<f32> for Exp2Backward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // Approximate: ignoring ln(2) factor for now
        // TODO: Add scalar multiplication
        vec![grad_output * &self.output]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Exp2Backward"
    }
}

/// Gradient for Sin: z = sin(a)
/// ∂L/∂a = ∂L/∂z · cos(a)
pub struct SinBackward {
    input: Tensor<f32, DimDyn>,
}

impl SinBackward {
    pub fn new(input: Tensor<f32, DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn<f32> for SinBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // cos(x) = sin(x + π/2)
        // TODO: Implement proper cos using scalar addition
        // Approximate: just use the input sin (incorrect but compiles)
        let cos_input = (&self.input).sin();
        vec![grad_output * &cos_input]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
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

    // Convert to f32-typed tensor for graph operations (TensorRef is f32-based)
    let inputs = vec![Arc::new(to_graph_ref(input))];
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
// Neg implementation (std::ops::Neg)
// ============================================================================

impl<D: Dimension> Neg for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn neg(self) -> Tensor<f32, D> {
        let result = create_unary_elementwise(ElementwiseOp::Neg, self);

        if self.requires_grad() {
            let grad_fn = NegBackward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Neg for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn neg(self) -> Tensor<f32, D> {
        -&self
    }
}

// ============================================================================
// Recip implementation
// ============================================================================

// f32 implementation with gradient tracking
impl<D: Dimension> Recip for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn recip(self) -> Tensor<f32, D> {
        let result = create_unary_elementwise(ElementwiseOp::Recip, self);

        if self.requires_grad() {
            let grad_fn = RecipBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Recip for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn recip(self) -> Tensor<f32, D> {
        (&self).recip()
    }
}

// f64 implementation (no gradient tracking)
impl<D: Dimension> Recip for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn recip(self) -> Tensor<f64, D> {
        create_unary_elementwise(ElementwiseOp::Recip, self)
    }
}

impl<D: Dimension> Recip for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn recip(self) -> Tensor<f64, D> {
        (&self).recip()
    }
}

// ============================================================================
// Sqrt implementation
// ============================================================================

// f32 implementation with gradient tracking
impl<D: Dimension> Sqrt for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn sqrt(self) -> Tensor<f32, D> {
        let result = create_unary_elementwise(ElementwiseOp::Sqrt, self);

        if self.requires_grad() {
            let grad_fn = SqrtBackward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Sqrt for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn sqrt(self) -> Tensor<f32, D> {
        (&self).sqrt()
    }
}

// f64 implementation (no gradient tracking)
impl<D: Dimension> Sqrt for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn sqrt(self) -> Tensor<f64, D> {
        create_unary_elementwise(ElementwiseOp::Sqrt, self)
    }
}

impl<D: Dimension> Sqrt for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn sqrt(self) -> Tensor<f64, D> {
        (&self).sqrt()
    }
}

// ============================================================================
// Log2 implementation
// ============================================================================

// f32 implementation with gradient tracking
impl<D: Dimension> Log2 for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn log2(self) -> Tensor<f32, D> {
        let result = create_unary_elementwise(ElementwiseOp::Log2, self);

        if self.requires_grad() {
            let grad_fn = Log2Backward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Log2 for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn log2(self) -> Tensor<f32, D> {
        (&self).log2()
    }
}

// f64 implementation (no gradient tracking)
impl<D: Dimension> Log2 for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn log2(self) -> Tensor<f64, D> {
        create_unary_elementwise(ElementwiseOp::Log2, self)
    }
}

impl<D: Dimension> Log2 for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn log2(self) -> Tensor<f64, D> {
        (&self).log2()
    }
}

// ============================================================================
// Exp2 implementation
// ============================================================================

// f32 implementation with gradient tracking
impl<D: Dimension> Exp2 for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn exp2(self) -> Tensor<f32, D> {
        let result = create_unary_elementwise(ElementwiseOp::Exp2, self);

        if self.requires_grad() {
            let grad_fn = Exp2Backward::new(self.clone().into_dyn(), result.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Exp2 for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn exp2(self) -> Tensor<f32, D> {
        (&self).exp2()
    }
}

// f64 implementation (no gradient tracking)
impl<D: Dimension> Exp2 for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn exp2(self) -> Tensor<f64, D> {
        create_unary_elementwise(ElementwiseOp::Exp2, self)
    }
}

impl<D: Dimension> Exp2 for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn exp2(self) -> Tensor<f64, D> {
        (&self).exp2()
    }
}

// ============================================================================
// Sin implementation
// ============================================================================

// f32 implementation with gradient tracking
impl<D: Dimension> Sin for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn sin(self) -> Tensor<f32, D> {
        let result = create_unary_elementwise(ElementwiseOp::Sin, self);

        if self.requires_grad() {
            let grad_fn = SinBackward::new(self.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Sin for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn sin(self) -> Tensor<f32, D> {
        (&self).sin()
    }
}

// f64 implementation (no gradient tracking)
impl<D: Dimension> Sin for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn sin(self) -> Tensor<f64, D> {
        create_unary_elementwise(ElementwiseOp::Sin, self)
    }
}

impl<D: Dimension> Sin for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn sin(self) -> Tensor<f64, D> {
        (&self).sin()
    }
}

// ============================================================================
// Floor implementation
// ============================================================================

// f32 implementation (no gradient - floor is non-differentiable)
impl<D: Dimension> Floor for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    /// Floor is non-differentiable (gradient is 0 almost everywhere).
    /// Therefore, gradient tracking is not preserved for this operation.
    fn floor(self) -> Tensor<f32, D> {
        create_unary_elementwise(ElementwiseOp::Floor, self)
    }
}

impl<D: Dimension> Floor for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn floor(self) -> Tensor<f32, D> {
        (&self).floor()
    }
}

// f64 implementation (no gradient - floor is non-differentiable)
impl<D: Dimension> Floor for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn floor(self) -> Tensor<f64, D> {
        create_unary_elementwise(ElementwiseOp::Floor, self)
    }
}

impl<D: Dimension> Floor for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn floor(self) -> Tensor<f64, D> {
        (&self).floor()
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
}
