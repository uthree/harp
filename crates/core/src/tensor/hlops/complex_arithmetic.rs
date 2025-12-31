//! Complex number arithmetic operations
//!
//! This module provides arithmetic operators for complex tensors:
//! - Add: Tensor + Tensor
//! - Sub: Tensor - Tensor = Add(a, Neg(b))
//! - Mul: Tensor * Tensor
//! - Div: Tensor / Tensor = Mul(a, Recip(b))
//! - Neg: -Tensor
//!
//! ## Gradient Support (Wirtinger Derivatives)
//!
//! All operations support automatic differentiation using Wirtinger calculus.
//! For a real-valued loss L and complex operation z = f(a, b):
//! - The gradient propagated is ∂L/∂z* (conjugate Wirtinger derivative)
//! - For holomorphic functions: grad_input = grad_output * conj(∂f/∂z)

use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::tensor::primops::complex::Conjugate;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    Complex32, Complex64, ComplexGradFn, Dimension, ElementwiseOp, NumericDType, Tensor,
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

/// Create a binary elementwise Tensor for complex types
fn create_complex_binary_elementwise<T: NumericDType, D: Dimension>(
    op: ElementwiseOp,
    lhs: &Tensor<T, D>,
    rhs: &Tensor<T, impl Dimension>,
) -> Tensor<T, D> {
    let result_shape = broadcast_shapes(lhs.shape(), rhs.shape());
    let view = view_from_shape(&result_shape);

    let inputs = vec![lhs.as_input_ref(), rhs.as_input_ref()];
    let expr = op.to_ast(2);

    let inner = TensorInner::new(
        TensorOp::elementwise(inputs, expr),
        view,
        result_shape,
        T::DTYPE,
    );

    Tensor {
        inner: Arc::new(inner),
        autograd_meta: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

/// Create a unary elementwise Tensor for complex types
fn create_complex_unary_elementwise<T: NumericDType, D: Dimension>(
    op: ElementwiseOp,
    input: &Tensor<T, D>,
) -> Tensor<T, D> {
    let view = view_from_shape(input.shape());
    let shape = input.shape().to_vec();

    let inputs = vec![input.as_input_ref()];
    let expr = op.to_ast(1);

    let inner = TensorInner::new(TensorOp::elementwise(inputs, expr), view, shape, T::DTYPE);

    Tensor {
        inner: Arc::new(inner),
        autograd_meta: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

/// Helper to compute result shape for binary operations with broadcasting
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
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

// ============================================================================
// Macro for implementing arithmetic operations for complex types
// ============================================================================

macro_rules! impl_complex_add {
    ($complex_type:ty, $backward_type:ident) => {
        impl<D: Dimension> Add for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn add(self, rhs: Self) -> Tensor<$complex_type, D> {
                let result = create_complex_binary_elementwise(ElementwiseOp::Add, self, rhs);
                if any_complex_requires_grad(self, rhs) {
                    let grad_fn = $backward_type::new(self.clone(), rhs.clone());
                    result.with_complex_grad_fn(Arc::new(grad_fn))
                } else {
                    result
                }
            }
        }

        impl<D: Dimension> Add<Tensor<$complex_type, D>> for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn add(self, rhs: Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                self + &rhs
            }
        }

        impl<D: Dimension> Add<&Tensor<$complex_type, D>> for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn add(self, rhs: &Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                &self + rhs
            }
        }

        impl<D: Dimension> Add for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn add(self, rhs: Self) -> Tensor<$complex_type, D> {
                &self + &rhs
            }
        }
    };
}

macro_rules! impl_complex_mul {
    ($complex_type:ty, $backward_type:ident) => {
        impl<D: Dimension> Mul for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn mul(self, rhs: Self) -> Tensor<$complex_type, D> {
                let result = create_complex_binary_elementwise(ElementwiseOp::Mul, self, rhs);
                if any_complex_requires_grad(self, rhs) {
                    let grad_fn = $backward_type::new(self.clone(), rhs.clone());
                    result.with_complex_grad_fn(Arc::new(grad_fn))
                } else {
                    result
                }
            }
        }

        impl<D: Dimension> Mul<Tensor<$complex_type, D>> for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn mul(self, rhs: Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                self * &rhs
            }
        }

        impl<D: Dimension> Mul<&Tensor<$complex_type, D>> for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn mul(self, rhs: &Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                &self * rhs
            }
        }

        impl<D: Dimension> Mul for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn mul(self, rhs: Self) -> Tensor<$complex_type, D> {
                &self * &rhs
            }
        }
    };
}

macro_rules! impl_complex_neg {
    ($complex_type:ty, $backward_type:ident) => {
        impl<D: Dimension> Neg for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn neg(self) -> Tensor<$complex_type, D> {
                let result = create_complex_unary_elementwise(ElementwiseOp::Neg, self);
                if self.requires_grad() {
                    let grad_fn = $backward_type::new(self.clone());
                    result.with_complex_grad_fn(Arc::new(grad_fn))
                } else {
                    result
                }
            }
        }

        impl<D: Dimension> Neg for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn neg(self) -> Tensor<$complex_type, D> {
                -&self
            }
        }
    };
}

macro_rules! impl_complex_sub {
    ($complex_type:ty) => {
        impl<D: Dimension> Sub for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn sub(self, rhs: Self) -> Tensor<$complex_type, D> {
                // a - b = a + (-b), gradients tracked through Add and Neg
                self + &(-rhs)
            }
        }

        impl<D: Dimension> Sub<Tensor<$complex_type, D>> for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn sub(self, rhs: Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                self - &rhs
            }
        }

        impl<D: Dimension> Sub<&Tensor<$complex_type, D>> for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn sub(self, rhs: &Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                &self - rhs
            }
        }

        impl<D: Dimension> Sub for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn sub(self, rhs: Self) -> Tensor<$complex_type, D> {
                &self - &rhs
            }
        }
    };
}

macro_rules! impl_complex_recip {
    ($complex_type:ty, $backward_type:ident) => {
        impl<D: Dimension> crate::tensor::primops::Recip for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn recip(self) -> Tensor<$complex_type, D> {
                let result = create_complex_unary_elementwise(ElementwiseOp::Recip, self);
                if self.requires_grad() {
                    let grad_fn = $backward_type::new(self.clone(), result.clone());
                    result.with_complex_grad_fn(Arc::new(grad_fn))
                } else {
                    result
                }
            }
        }

        impl<D: Dimension> crate::tensor::primops::Recip for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn recip(self) -> Tensor<$complex_type, D> {
                (&self).recip()
            }
        }
    };
}

macro_rules! impl_complex_div {
    ($complex_type:ty) => {
        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<D: Dimension> Div for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn div(self, rhs: Self) -> Tensor<$complex_type, D> {
                // a / b = a * (1/b)
                use crate::tensor::primops::Recip;
                self * &rhs.recip()
            }
        }

        impl<D: Dimension> Div<Tensor<$complex_type, D>> for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn div(self, rhs: Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                self / &rhs
            }
        }

        impl<D: Dimension> Div<&Tensor<$complex_type, D>> for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn div(self, rhs: &Tensor<$complex_type, D>) -> Tensor<$complex_type, D> {
                &self / rhs
            }
        }

        impl<D: Dimension> Div for Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn div(self, rhs: Self) -> Tensor<$complex_type, D> {
                &self / &rhs
            }
        }
    };
}

// ============================================================================
// Apply macros for Complex32 and Complex64
// ============================================================================

impl_complex_add!(Complex32, Complex32AddBackward);
impl_complex_add!(Complex64, Complex64AddBackward);

impl_complex_mul!(Complex32, Complex32MulBackward);
impl_complex_mul!(Complex64, Complex64MulBackward);

impl_complex_neg!(Complex32, Complex32NegBackward);
impl_complex_neg!(Complex64, Complex64NegBackward);

impl_complex_sub!(Complex32);
impl_complex_sub!(Complex64);

impl_complex_recip!(Complex32, Complex32RecipBackward);
impl_complex_recip!(Complex64, Complex64RecipBackward);

impl_complex_div!(Complex32);
impl_complex_div!(Complex64);

// ============================================================================
// Gradient Functions (Wirtinger Derivatives)
// ============================================================================

/// Macro to implement gradient backward structs for complex binary operations
macro_rules! impl_complex_backward {
    ($name:ident, $complex_type:ty, $real_type:ty, $backward_impl:expr) => {
        pub struct $name<D: Dimension> {
            lhs: Tensor<$complex_type, D>,
            rhs: Tensor<$complex_type, D>,
        }

        impl<D: Dimension> $name<D> {
            pub fn new(lhs: Tensor<$complex_type, D>, rhs: Tensor<$complex_type, D>) -> Self {
                Self { lhs, rhs }
            }
        }

        impl<D: Dimension> ComplexGradFn<$real_type, D> for $name<D> {
            fn backward(&self, grad_output: &Tensor<$complex_type, D>) {
                $backward_impl(self, grad_output);
            }

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

/// Macro to implement gradient backward structs for complex unary operations
macro_rules! impl_complex_unary_backward {
    ($name:ident, $complex_type:ty, $real_type:ty, $backward_impl:expr) => {
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
                $backward_impl(self, grad_output);
            }

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

// Add backward: grad_a = grad_out, grad_b = grad_out
impl_complex_backward!(
    Complex32AddBackward,
    Complex32,
    f32,
    |this: &Complex32AddBackward<D>, grad_output: &Tensor<Complex32, D>| {
        if this.lhs.requires_grad() {
            this.lhs.backward_with(grad_output.clone());
        }
        if this.rhs.requires_grad() {
            this.rhs.backward_with(grad_output.clone());
        }
    }
);

impl_complex_backward!(
    Complex64AddBackward,
    Complex64,
    f64,
    |this: &Complex64AddBackward<D>, grad_output: &Tensor<Complex64, D>| {
        if this.lhs.requires_grad() {
            this.lhs.backward_with(grad_output.clone());
        }
        if this.rhs.requires_grad() {
            this.rhs.backward_with(grad_output.clone());
        }
    }
);

// Mul backward: grad_a = grad_out * conj(b), grad_b = grad_out * conj(a)
impl_complex_backward!(
    Complex32MulBackward,
    Complex32,
    f32,
    |this: &Complex32MulBackward<D>, grad_output: &Tensor<Complex32, D>| {
        if this.lhs.requires_grad() {
            let grad_lhs = grad_output * &this.rhs.conj();
            this.lhs.backward_with(grad_lhs);
        }
        if this.rhs.requires_grad() {
            let grad_rhs = grad_output * &this.lhs.conj();
            this.rhs.backward_with(grad_rhs);
        }
    }
);

impl_complex_backward!(
    Complex64MulBackward,
    Complex64,
    f64,
    |this: &Complex64MulBackward<D>, grad_output: &Tensor<Complex64, D>| {
        if this.lhs.requires_grad() {
            let grad_lhs = grad_output * &this.rhs.conj();
            this.lhs.backward_with(grad_lhs);
        }
        if this.rhs.requires_grad() {
            let grad_rhs = grad_output * &this.lhs.conj();
            this.rhs.backward_with(grad_rhs);
        }
    }
);

// Neg backward: grad_a = -grad_out
impl_complex_unary_backward!(
    Complex32NegBackward,
    Complex32,
    f32,
    |this: &Complex32NegBackward<D>, grad_output: &Tensor<Complex32, D>| {
        if this.input.requires_grad() {
            this.input.backward_with(-grad_output);
        }
    }
);

impl_complex_unary_backward!(
    Complex64NegBackward,
    Complex64,
    f64,
    |this: &Complex64NegBackward<D>, grad_output: &Tensor<Complex64, D>| {
        if this.input.requires_grad() {
            this.input.backward_with(-grad_output);
        }
    }
);

// Recip backward: z = 1/a
// f'(a) = -1/a² = -(1/a)² = -z²
// grad_a = grad_out * conj(-z²) = -grad_out * conj(z)²
/// Recip backward struct that stores both input and output
macro_rules! impl_complex_recip_backward {
    ($name:ident, $complex_type:ty, $real_type:ty) => {
        pub struct $name<D: Dimension> {
            input: Tensor<$complex_type, D>,
            output: Tensor<$complex_type, D>,
        }

        impl<D: Dimension> $name<D> {
            pub fn new(input: Tensor<$complex_type, D>, output: Tensor<$complex_type, D>) -> Self {
                Self { input, output }
            }
        }

        impl<D: Dimension> ComplexGradFn<$real_type, D> for $name<D> {
            fn backward(&self, grad_output: &Tensor<$complex_type, D>) {
                if self.input.requires_grad() {
                    // grad = -grad_out * conj(output)²
                    let output_conj = self.output.conj();
                    let output_conj_sq = &output_conj * &output_conj;
                    let grad = -(grad_output * &output_conj_sq);
                    self.input.backward_with(grad);
                }
            }

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

impl_complex_recip_backward!(Complex32RecipBackward, Complex32, f32);
impl_complex_recip_backward!(Complex64RecipBackward, Complex64, f64);

// ============================================================================
// Helper function to check if any input requires grad
// ============================================================================

fn any_complex_requires_grad<T: NumericDType, D: Dimension>(
    lhs: &Tensor<T, D>,
    rhs: &Tensor<T, D>,
) -> bool {
    lhs.requires_grad() || rhs.requires_grad()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_complex32_add() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(3.0f32, 4.0f32));
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex32_mul() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(3.0f32, 4.0f32));
        let c = &a * &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex32_neg() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
        let c = -&a;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex32_sub() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(5.0f32, 6.0f32));
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
        let c = &a - &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex32_div() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(6.0f32, 8.0f32));
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(2.0f32, 0.0f32));
        let c = &a / &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex64_add() {
        let a = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(1.0f64, 2.0f64));
        let b = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(3.0f64, 4.0f64));
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex64_mul() {
        let a = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(1.0f64, 2.0f64));
        let b = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(3.0f64, 4.0f64));
        let c = &a * &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex64_neg() {
        let a = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(1.0f64, 2.0f64));
        let c = -&a;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex64_sub() {
        let a = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(5.0f64, 6.0f64));
        let b = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(1.0f64, 2.0f64));
        let c = &a - &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex64_div() {
        let a = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(6.0f64, 8.0f64));
        let b = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(2.0f64, 0.0f64));
        let c = &a / &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_complex_add_owned() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(3.0f32, 4.0f32));
        let c = a + b; // owned + owned
        assert_eq!(c.shape(), &[2, 3]);
    }

    // ============================================================================
    // Autograd tests
    // ============================================================================

    use crate::tensor::Complex;

    #[test]
    fn test_complex32_requires_grad() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
        assert!(!a.requires_grad());

        let a_grad = a.set_requires_grad(true);
        assert!(a_grad.requires_grad());
    }

    #[test]
    fn test_complex32_add_tracks_grad() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32))
            .set_requires_grad(true);
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(3.0f32, 4.0f32))
            .set_requires_grad(true);
        let c = &a + &b;

        // Result should also require grad since inputs require grad
        assert!(c.requires_grad());
    }

    #[test]
    fn test_complex32_mul_tracks_grad() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32))
            .set_requires_grad(true);
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(3.0f32, 4.0f32))
            .set_requires_grad(true);
        let c = &a * &b;

        assert!(c.requires_grad());
    }

    #[test]
    fn test_complex32_neg_tracks_grad() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32))
            .set_requires_grad(true);
        let c = -&a;

        assert!(c.requires_grad());
    }

    #[test]
    fn test_complex32_sub_tracks_grad() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(5.0f32, 6.0f32))
            .set_requires_grad(true);
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32))
            .set_requires_grad(true);
        let c = &a - &b;

        assert!(c.requires_grad());
    }

    #[test]
    fn test_complex32_div_tracks_grad() {
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(6.0f32, 8.0f32))
            .set_requires_grad(true);
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(2.0f32, 1.0f32))
            .set_requires_grad(true);
        let c = &a / &b;

        assert!(c.requires_grad());
    }

    #[test]
    fn test_complex32_chain_tracks_grad() {
        // Test a chain of operations: (a + b) * c
        let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32))
            .set_requires_grad(true);
        let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(3.0f32, 4.0f32))
            .set_requires_grad(true);
        let c = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(0.5f32, 0.5f32))
            .set_requires_grad(true);

        let sum = &a + &b;
        let result = &sum * &c;

        assert!(result.requires_grad());
    }

    #[test]
    fn test_complex64_autograd_basic() {
        let a = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(1.0f64, 2.0f64))
            .set_requires_grad(true);
        let b = Tensor::<Complex64, Dim2>::full([2, 3], Complex::new(3.0f64, 4.0f64))
            .set_requires_grad(true);

        let c = &a + &b;
        let d = &c * &a;

        assert!(d.requires_grad());
    }
}
