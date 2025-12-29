//! Initialization operations (primops)
//!
//! - Const: constant tensor
//! - Rand: uniform random [0, 1)

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::RwLock;

use crate::ast::DType;
use crate::backend::Buffer;
use crate::tensor::forward::VecBuffer;
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    Dim, DimDyn, Dimension, FloatDType, NumericDType, Tensor, TensorInner, TensorOp,
};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

/// Generate random f32 values in [0, 1)
///
/// Uses a simple Xorshift PRNG for fast random number generation.
/// Thread-local state ensures thread safety without locks.
fn generate_random_f32(count: usize) -> Vec<f32> {
    use std::cell::RefCell;

    thread_local! {
        static STATE: RefCell<u64> = const { RefCell::new(0) };
    }

    // Initialize seed if needed
    STATE.with(|state| {
        let mut s = state.borrow_mut();
        if *s == 0 {
            // Use current time as seed
            *s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x12345678deadbeef);
        }
    });

    let mut result = Vec::with_capacity(count);
    STATE.with(|state| {
        let mut s = state.borrow_mut();
        for _ in 0..count {
            // Xorshift64
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            // Convert to [0, 1) range
            let val = (*s as f32) / (u64::MAX as f32);
            result.push(val);
        }
    });
    result
}

/// Generate random f64 values in [0, 1)
fn generate_random_f64(count: usize) -> Vec<f64> {
    use std::cell::RefCell;

    thread_local! {
        static STATE: RefCell<u64> = const { RefCell::new(0) };
    }

    // Initialize seed if needed
    STATE.with(|state| {
        let mut s = state.borrow_mut();
        if *s == 0 {
            *s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x12345678deadbeef);
        }
    });

    let mut result = Vec::with_capacity(count);
    STATE.with(|state| {
        let mut s = state.borrow_mut();
        for _ in 0..count {
            // Xorshift64
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            // Convert to [0, 1) range
            let val = (*s as f64) / (u64::MAX as f64);
            result.push(val);
        }
    });
    result
}

// ============================================================================
// Static dimension constructors (NumericDType - all numeric types)
// ============================================================================

impl<T: NumericDType, const N: usize> Tensor<T, Dim<N>>
where
    Dim<N>: Dimension,
{
    /// Create a tensor filled with zeros (Const(0))
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::full(shape, T::ZERO)
    }

    /// Create a tensor filled with ones (Const(1))
    pub fn ones(shape: [usize; N]) -> Self {
        Self::full(shape, T::ONE)
    }

    /// Create a tensor filled with a constant value
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

// ============================================================================
// rand - FloatDType only (f32, f64)
// ============================================================================

impl<T: FloatDType, const N: usize> Tensor<T, Dim<N>>
where
    Dim<N>: Dimension,
{
    /// Create a tensor with uniform random values [0, 1)
    ///
    /// Random values are generated on the CPU and stored in a buffer.
    /// This ensures compatibility with all backends (OpenCL, Metal, etc.).
    pub fn rand(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let numel: usize = shape_vec.iter().product();
        let view = view_from_shape(&shape_vec);

        // Generate random data based on dtype
        let vec_buffer: Box<dyn Buffer> = match T::DTYPE {
            DType::F32 => {
                let data = generate_random_f32(numel);
                Box::new(VecBuffer::from_vec(&data, shape_vec.clone(), DType::F32))
            }
            DType::F64 => {
                let data = generate_random_f64(numel);
                Box::new(VecBuffer::from_vec(&data, shape_vec.clone(), DType::F64))
            }
            _ => unreachable!("FloatDType only supports F32 and F64"),
        };

        let inner = TensorInner {
            op: TensorOp::Executed,
            view,
            shape: shape_vec,
            dtype: T::DTYPE,
            name: None,
            autograd: None,
            buffer: RwLock::new(Some(vec_buffer)),
        };
        Self {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Dynamic dimension constructors (generic over NumericDType)
// ============================================================================

impl<T: NumericDType> Tensor<T, DimDyn> {
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
    ///
    /// Random values are generated on the CPU and stored in a buffer.
    /// This ensures compatibility with all backends (OpenCL, Metal, etc.).
    pub fn rand_dyn(shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let numel: usize = shape_vec.iter().product();
        let view = view_from_shape(&shape_vec);

        // Generate random data based on dtype
        let vec_buffer: Box<dyn Buffer> = match T::DTYPE {
            DType::F32 => {
                let data = generate_random_f32(numel);
                Box::new(VecBuffer::from_vec(&data, shape_vec.clone(), DType::F32))
            }
            DType::F64 => {
                let data = generate_random_f64(numel);
                Box::new(VecBuffer::from_vec(&data, shape_vec.clone(), DType::F64))
            }
            _ => unreachable!("FloatDType only supports F32 and F64"),
        };

        let inner = TensorInner {
            op: TensorOp::Executed,
            view,
            shape: shape_vec,
            dtype: T::DTYPE,
            name: None,
            autograd: None,
            buffer: RwLock::new(Some(vec_buffer)),
        };
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
