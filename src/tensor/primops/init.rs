//! Initialization operations (primops)
//!
//! - Const: constant tensor
//! - Rand: uniform random [0, 1)

use std::marker::PhantomData;
use std::sync::Arc;

use crate::ast::DType;
use crate::ast::Literal;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    Dim, DimDyn, Dimension, FloatDType, IntegerDType, NumericInitDType, Tensor, TensorInner,
    TensorOp,
};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

// ============================================================================
// Static dimension constructors
// ============================================================================

impl<const N: usize> Tensor<f32, Dim<N>>
where
    Dim<N>: Dimension,
{
    /// Create a tensor filled with zeros (Const(0))
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::full(shape, 0.0)
    }

    /// Create a tensor filled with ones (Const(1))
    pub fn ones(shape: [usize; N]) -> Self {
        Self::full(shape, 1.0)
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: [usize; N], value: f32) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new(
            TensorOp::ConstFill(Literal::F32(value)),
            view,
            shape_vec,
            DType::F32,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Create an input tensor (placeholder for data)
    pub fn input(name: &str, shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            view,
            shape_vec,
            DType::F32,
            name,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Create a tensor with uniform random values [0, 1)
    pub fn rand(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new(TensorOp::Rand, view, shape_vec, DType::F32);
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Static dimension constructors (f64)
// ============================================================================

impl<const N: usize> Tensor<f64, Dim<N>>
where
    Dim<N>: Dimension,
{
    /// Create a tensor filled with zeros (Const(0))
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::full(shape, 0.0)
    }

    /// Create a tensor filled with ones (Const(1))
    pub fn ones(shape: [usize; N]) -> Self {
        Self::full(shape, 1.0)
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: [usize; N], value: f64) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new(
            TensorOp::ConstFill(Literal::F64(value)),
            view,
            shape_vec,
            DType::F64,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Create an input tensor (placeholder for data)
    pub fn input(name: &str, shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            view,
            shape_vec,
            DType::F64,
            name,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Create a tensor with uniform random values [0, 1)
    pub fn rand(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new(TensorOp::Rand, view, shape_vec, DType::F64);
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Dynamic dimension constructors (generic over NumericInitDType)
// ============================================================================

impl<T: NumericInitDType> Tensor<T, DimDyn> {
    /// Create a tensor filled with zeros (dynamic shape)
    pub fn zeros_dyn(shape: &[usize]) -> Self {
        Self::full_dyn(shape, T::ZERO)
    }

    /// Create a tensor filled with ones (dynamic shape)
    pub fn ones_dyn(shape: &[usize]) -> Self {
        Self::full_dyn(shape, T::ONE)
    }

    /// Create a tensor filled with a constant value (dynamic shape)
    pub fn full_dyn(shape: &[usize], value: T) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new(
            TensorOp::ConstFill(T::to_literal(value)),
            view,
            shape_vec,
            T::DTYPE,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Create an input tensor (dynamic shape)
    pub fn input_dyn(name: &str, shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            view,
            shape_vec,
            T::DTYPE,
            name,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// rand_dyn is only available for FloatDType
impl<T: FloatDType> Tensor<T, DimDyn> {
    /// Create a tensor with uniform random values [0, 1) (dynamic shape)
    pub fn rand_dyn(shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new(TensorOp::Rand, view, shape_vec, T::DTYPE);
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Static dimension constructors (generic over IntegerDType)
// ============================================================================

impl<T: IntegerDType, const N: usize> Tensor<T, Dim<N>>
where
    Dim<N>: Dimension,
{
    /// Create an integer tensor filled with zeros
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::full(shape, T::ZERO)
    }

    /// Create an integer tensor filled with ones
    pub fn ones(shape: [usize; N]) -> Self {
        Self::full(shape, T::ONE)
    }

    /// Create an integer tensor filled with a constant value
    pub fn full(shape: [usize; N], value: T) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new(
            TensorOp::ConstFill(T::to_literal(value)),
            view,
            shape_vec,
            T::DTYPE,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Create an integer input tensor
    pub fn input(name: &str, shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let inner = TensorInner::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            view,
            shape_vec,
            T::DTYPE,
            name,
        );
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim1, Dim2, Dim3};

    #[test]
    fn test_zeros() {
        let t = Tensor::<f32, Dim2>::zeros([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_ones() {
        let t = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_full() {
        let t = Tensor::<f32, Dim1>::full([10], 2.5);
        assert_eq!(t.shape(), &[10]);
    }

    #[test]
    fn test_rand() {
        let t = Tensor::<f32, Dim2>::rand([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_zeros_dyn() {
        let t = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
    }

    #[test]
    fn test_rand_dyn() {
        let t = Tensor::<f32, DimDyn>::rand_dyn(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    // f64 tests
    #[test]
    fn test_zeros_f64() {
        let t = Tensor::<f64, Dim2>::zeros([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_ones_f64() {
        let t = Tensor::<f64, Dim3>::ones([2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_full_f64() {
        let t = Tensor::<f64, Dim1>::full([10], 2.5);
        assert_eq!(t.shape(), &[10]);
    }

    #[test]
    fn test_zeros_dyn_f64() {
        let t = Tensor::<f64, DimDyn>::zeros_dyn(&[3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
    }

    // Integer tests
    #[test]
    fn test_zeros_i32() {
        let t = Tensor::<i32, Dim2>::zeros([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_ones_i64() {
        let t = Tensor::<i64, Dim2>::ones([2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
    }

    #[test]
    fn test_full_u32() {
        let t = Tensor::<u32, Dim1>::full([10], 42);
        assert_eq!(t.shape(), &[10]);
    }

    #[test]
    fn test_zeros_dyn_u64() {
        let t = Tensor::<u64, DimDyn>::zeros_dyn(&[3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
    }

    #[test]
    fn test_full_dyn_i8() {
        let t = Tensor::<i8, DimDyn>::full_dyn(&[2, 3], 5);
        assert_eq!(t.shape(), &[2, 3]);
    }
}
