//! Complex number arithmetic operations
//!
//! This module provides arithmetic operators for complex tensors:
//! - Add: Tensor + Tensor
//! - Sub: Tensor - Tensor = Add(a, Neg(b))
//! - Mul: Tensor * Tensor
//! - Div: Tensor / Tensor = Mul(a, Recip(b))
//! - Neg: -Tensor
//!
//! Note: Gradient tracking for complex numbers is not yet implemented.
//! It will be added in the "勾配関数追加" task using Wirtinger derivatives.

use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    Complex32, Complex64, Dimension, ElementwiseOp, NumericDType, Tensor, TensorInner, TensorOp,
};

#[cfg(test)]
use crate::tensor::Complex;

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
    ($complex_type:ty) => {
        impl<D: Dimension> Add for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn add(self, rhs: Self) -> Tensor<$complex_type, D> {
                create_complex_binary_elementwise(ElementwiseOp::Add, self, rhs)
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
    ($complex_type:ty) => {
        impl<D: Dimension> Mul for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn mul(self, rhs: Self) -> Tensor<$complex_type, D> {
                create_complex_binary_elementwise(ElementwiseOp::Mul, self, rhs)
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
    ($complex_type:ty) => {
        impl<D: Dimension> Neg for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn neg(self) -> Tensor<$complex_type, D> {
                create_complex_unary_elementwise(ElementwiseOp::Neg, self)
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
                // a - b = a + (-b)
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
    ($complex_type:ty) => {
        impl<D: Dimension> crate::tensor::primops::Recip for &Tensor<$complex_type, D> {
            type Output = Tensor<$complex_type, D>;
            fn recip(self) -> Tensor<$complex_type, D> {
                create_complex_unary_elementwise(ElementwiseOp::Recip, self)
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

impl_complex_add!(Complex32);
impl_complex_add!(Complex64);

impl_complex_mul!(Complex32);
impl_complex_mul!(Complex64);

impl_complex_neg!(Complex32);
impl_complex_neg!(Complex64);

impl_complex_sub!(Complex32);
impl_complex_sub!(Complex64);

impl_complex_recip!(Complex32);
impl_complex_recip!(Complex64);

impl_complex_div!(Complex32);
impl_complex_div!(Complex64);

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
}
