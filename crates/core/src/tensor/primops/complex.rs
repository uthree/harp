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
use crate::tensor::{
    Complex, Complex32, Complex64, ComplexDType, ComplexGradFn, Dimension, FloatDType, Tensor,
    TensorInner, TensorOp,
};

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

// Conjugate backward: conj is non-holomorphic
// ∂/∂z conj(z) = 0, ∂/∂z* conj(z) = 1
// For Wirtinger derivatives: grad_input = conj(grad_output)
macro_rules! impl_conj_backward {
    ($name:ident, $complex_type:ty, $real_type:ty) => {
        pub struct $name<D: Dimension> {
            input: Tensor<$complex_type, D>,
        }

        impl<D: Dimension> $name<D> {
            pub fn new(input: Tensor<$complex_type, D>) -> Self {
                Self { input }
            }
        }

        impl<D: Dimension> ComplexGradFn<$real_type, D> for $name<D> {
            fn backward(&self, grad_output: &Tensor<$complex_type, D>) {
                if self.input.requires_grad() {
                    // conj is non-holomorphic: grad = conj(grad_output)
                    self.input.backward_with(grad_output.conj());
                }
            }

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

impl_conj_backward!(Complex32ConjBackward, Complex32, f32);
impl_conj_backward!(Complex64ConjBackward, Complex64, f64);

macro_rules! impl_complex_conj {
    ($complex_type:ty, $real_type:ty, $backward_type:ident) => {
        impl<D: Dimension> Conjugate for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;

            fn conj(&self) -> Self::Output {
                let view = view_from_shape(self.shape());
                let shape = self.shape().to_vec();

                let inputs = vec![self.as_input_ref()];
                let expr = conj(wildcard("0"));

                let inner = TensorInner::new(
                    TensorOp::elementwise(inputs, expr),
                    view,
                    shape,
                    <$complex_type as TensorDType>::DTYPE,
                );

                let result: Tensor<$complex_type, D> = Tensor {
                    inner: Arc::new(inner),
                    autograd_meta: None,
                    _dtype: PhantomData,
                    _dim: PhantomData,
                };

                if self.requires_grad() {
                    let grad_fn: Arc<dyn ComplexGradFn<$real_type, D>> =
                        Arc::new($backward_type::new(self.clone()));
                    result.with_complex_grad_fn(grad_fn)
                } else {
                    result
                }
            }
        }
    };
}

impl_complex_conj!(Complex32, f32, Complex32ConjBackward);
impl_complex_conj!(Complex64, f64, Complex64ConjBackward);

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
