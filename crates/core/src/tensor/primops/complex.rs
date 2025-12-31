//! Complex number tensor operations
//!
//! This module provides primitive operations for complex tensors:
//!
//! - `real()` - Extract real part: `Tensor<Complex<T>, D>` → `Tensor<T, D>`
//! - `imag()` - Extract imaginary part: `Tensor<Complex<T>, D>` → `Tensor<T, D>`
//! - `conj()` - Complex conjugate: `Tensor<Complex<T>, D>` → `Tensor<Complex<T>, D>`
//! - `complex()` - Construct complex tensor from real and imaginary parts

use std::marker::PhantomData;
use std::sync::Arc;

use crate::ast::helper::{conj, imag, make_complex, real, wildcard};
use crate::tensor::dtype::TensorDType;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{Complex, ComplexDType, Dimension, FloatDType, Tensor, TensorInner, TensorOp};

// ============================================================================
// Helper functions
// ============================================================================

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

// ============================================================================
// Complex part extraction traits
// ============================================================================

/// Trait for extracting the real part of a complex tensor
pub trait RealPart {
    type Output;
    fn real(&self) -> Self::Output;
}

/// Trait for extracting the imaginary part of a complex tensor
pub trait ImagPart {
    type Output;
    fn imag(&self) -> Self::Output;
}

/// Trait for complex conjugate
pub trait Conjugate {
    type Output;
    fn conj(&self) -> Self::Output;
}

// ============================================================================
// Real part extraction
// ============================================================================

impl<T: ComplexDType, D: Dimension> RealPart for Tensor<T, D>
where
    T::Real: TensorDType,
{
    type Output = Tensor<T::Real, D>;

    fn real(&self) -> Self::Output {
        let view = view_from_shape(self.shape());
        let shape = self.shape().to_vec();

        let inputs = vec![self.as_input_ref()];
        let expr = real(wildcard("0"));

        let inner = TensorInner::new(
            TensorOp::elementwise(inputs, expr),
            view,
            shape,
            <T::Real as TensorDType>::DTYPE,
        );

        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Imaginary part extraction
// ============================================================================

impl<T: ComplexDType, D: Dimension> ImagPart for Tensor<T, D>
where
    T::Real: TensorDType,
{
    type Output = Tensor<T::Real, D>;

    fn imag(&self) -> Self::Output {
        let view = view_from_shape(self.shape());
        let shape = self.shape().to_vec();

        let inputs = vec![self.as_input_ref()];
        let expr = imag(wildcard("0"));

        let inner = TensorInner::new(
            TensorOp::elementwise(inputs, expr),
            view,
            shape,
            <T::Real as TensorDType>::DTYPE,
        );

        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Complex conjugate
// ============================================================================

impl<T: ComplexDType, D: Dimension> Conjugate for Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn conj(&self) -> Self::Output {
        let view = view_from_shape(self.shape());
        let shape = self.shape().to_vec();

        let inputs = vec![self.as_input_ref()];
        let expr = conj(wildcard("0"));

        let inner = TensorInner::new(TensorOp::elementwise(inputs, expr), view, shape, T::DTYPE);

        // TODO: Add gradient support using Wirtinger derivatives
        // For now, we don't track gradients for complex conjugate
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Complex construction from parts
// ============================================================================

/// Construct a complex tensor from real and imaginary parts
///
/// # Example
///
/// ```ignore
/// use harp_core::tensor::{Tensor, Dim2, Complex32};
/// use harp_core::tensor::primops::complex::complex;
///
/// let re = Tensor::<f32, Dim2>::ones([2, 3]);
/// let im = Tensor::<f32, Dim2>::zeros([2, 3]);
/// let z: Tensor<Complex32, Dim2> = complex(&re, &im);
/// ```
pub fn complex<T: FloatDType, D: Dimension>(
    real_part: &Tensor<T, D>,
    imag_part: &Tensor<T, D>,
) -> Tensor<Complex<T>, D>
where
    Complex<T>: ComplexDType<Real = T>,
{
    assert_eq!(
        real_part.shape(),
        imag_part.shape(),
        "Real and imaginary parts must have the same shape"
    );

    let view = view_from_shape(real_part.shape());
    let shape = real_part.shape().to_vec();

    let inputs = vec![real_part.as_input_ref(), imag_part.as_input_ref()];
    let expr = make_complex(wildcard("0"), wildcard("1"));

    let inner = TensorInner::new(
        TensorOp::elementwise(inputs, expr),
        view,
        shape,
        Complex::<T>::DTYPE,
    );

    Tensor {
        inner: Arc::new(inner),
        autograd_meta: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Tests will be added when tensor execution is fully implemented for complex types
}
