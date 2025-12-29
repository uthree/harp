//! Bitwise primitive operations (IntegerDType only)
//!
//! - BitAnd: bitwise AND
//! - BitOr: bitwise OR
//! - BitXor: bitwise XOR
//! - BitNot: bitwise NOT
//! - Shl: left shift
//! - Shr: right shift

use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::shape::{Expr, View};
use crate::tensor::{Dimension, ElementwiseOp, IntegerDType, Tensor, TensorInner, TensorOp};

use super::binary::broadcast_shapes;

// ============================================================================
// Helper functions
// ============================================================================

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

/// Create a binary bitwise Tensor using Compute variant
fn create_binary_bitwise<T: IntegerDType, D: Dimension>(
    op: ElementwiseOp,
    lhs: &Tensor<T, D>,
    rhs: &Tensor<T, impl Dimension>,
) -> Tensor<T, D> {
    let result_shape = broadcast_shapes(lhs.shape(), rhs.shape());
    let view = view_from_shape(&result_shape);

    // Create Compute operation with inputs embedded
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
        autograd_typed: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

/// Create a unary bitwise Tensor using Compute variant
fn create_unary_bitwise<T: IntegerDType, D: Dimension>(
    op: ElementwiseOp,
    input: &Tensor<T, D>,
) -> Tensor<T, D> {
    let view = view_from_shape(input.shape());

    // Create Compute operation with input embedded
    let inputs = vec![input.as_input_ref()];
    let expr = op.to_ast(1);

    let inner = TensorInner::new(
        TensorOp::elementwise(inputs, expr),
        view,
        input.shape().to_vec(),
        T::DTYPE,
    );

    Tensor {
        inner: Arc::new(inner),
        autograd_typed: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

/// Create a scalar tensor for shift operations
fn scalar_tensor<T: IntegerDType>(value: T) -> Tensor<T, crate::tensor::DimDyn> {
    let view = View::contiguous(Vec::<Expr>::new());
    let inner = TensorInner::new(
        TensorOp::ConstFill(T::to_literal(value)),
        view,
        vec![],
        T::DTYPE,
    );
    Tensor {
        inner: Arc::new(inner),
        autograd_typed: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Bitwise operations for IntegerDType
// ============================================================================

impl<T: IntegerDType, D: Dimension> Tensor<T, D> {
    /// Bitwise AND
    ///
    /// Computes element-wise bitwise AND: `self & other`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
    /// let b = Tensor::<i32, Dim2>::full([2, 3], 0b1010);
    /// let c = a.bitand(&b); // Results in 0b1000
    /// ```
    pub fn bitand(&self, other: &Tensor<T, impl Dimension>) -> Tensor<T, D> {
        create_binary_bitwise(ElementwiseOp::BitAnd, self, other)
    }

    /// Bitwise OR
    ///
    /// Computes element-wise bitwise OR: `self | other`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
    /// let b = Tensor::<i32, Dim2>::full([2, 3], 0b1010);
    /// let c = a.bitor(&b); // Results in 0b1110
    /// ```
    pub fn bitor(&self, other: &Tensor<T, impl Dimension>) -> Tensor<T, D> {
        create_binary_bitwise(ElementwiseOp::BitOr, self, other)
    }

    /// Bitwise XOR
    ///
    /// Computes element-wise bitwise XOR: `self ^ other`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
    /// let b = Tensor::<i32, Dim2>::full([2, 3], 0b1010);
    /// let c = a.bitxor(&b); // Results in 0b0110
    /// ```
    pub fn bitxor(&self, other: &Tensor<T, impl Dimension>) -> Tensor<T, D> {
        create_binary_bitwise(ElementwiseOp::BitXor, self, other)
    }

    /// Bitwise NOT
    ///
    /// Computes element-wise bitwise NOT: `!self`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
    /// let c = a.bitnot(); // Results in ~0b1100
    /// ```
    pub fn bitnot(&self) -> Tensor<T, D> {
        create_unary_bitwise(ElementwiseOp::BitNot, self)
    }

    /// Left shift
    ///
    /// Computes element-wise left shift: `self << other`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 1);
    /// let b = Tensor::<i32, Dim2>::full([2, 3], 3);
    /// let c = a.shl(&b); // Results in 8 (1 << 3)
    /// ```
    pub fn shl(&self, other: &Tensor<T, impl Dimension>) -> Tensor<T, D> {
        create_binary_bitwise(ElementwiseOp::Shl, self, other)
    }

    /// Right shift
    ///
    /// Computes element-wise right shift: `self >> other`
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 8);
    /// let b = Tensor::<i32, Dim2>::full([2, 3], 2);
    /// let c = a.shr(&b); // Results in 2 (8 >> 2)
    /// ```
    pub fn shr(&self, other: &Tensor<T, impl Dimension>) -> Tensor<T, D> {
        create_binary_bitwise(ElementwiseOp::Shr, self, other)
    }

    /// Left shift by scalar
    ///
    /// Computes element-wise left shift by a constant: `self << bits`
    pub fn shl_scalar(&self, bits: T) -> Tensor<T, D> {
        let scalar = scalar_tensor(bits);
        create_binary_bitwise(ElementwiseOp::Shl, self, &scalar)
    }

    /// Right shift by scalar
    ///
    /// Computes element-wise right shift by a constant: `self >> bits`
    pub fn shr_scalar(&self, bits: T) -> Tensor<T, D> {
        let scalar = scalar_tensor(bits);
        create_binary_bitwise(ElementwiseOp::Shr, self, &scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_bitand_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
        let b = Tensor::<i32, Dim2>::full([2, 3], 0b1010);
        let c = a.bitand(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_bitor_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
        let b = Tensor::<i32, Dim2>::full([2, 3], 0b1010);
        let c = a.bitor(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_bitxor_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
        let b = Tensor::<i32, Dim2>::full([2, 3], 0b1010);
        let c = a.bitxor(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_bitnot_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 0b1100);
        let c = a.bitnot();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_shl_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 1);
        let b = Tensor::<i32, Dim2>::full([2, 3], 3);
        let c = a.shl(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_shr_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 8);
        let b = Tensor::<i32, Dim2>::full([2, 3], 2);
        let c = a.shr(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_shl_scalar_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 1);
        let c = a.shl_scalar(3);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_shr_scalar_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 8);
        let c = a.shr_scalar(2);
        assert_eq!(c.shape(), &[2, 3]);
    }

    // Test with u64 type
    #[test]
    fn test_bitand_u64() {
        let a = Tensor::<u64, Dim2>::full([2, 3], 0xFF00);
        let b = Tensor::<u64, Dim2>::full([2, 3], 0x0FF0);
        let c = a.bitand(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_bitor_u64() {
        let a = Tensor::<u64, Dim2>::full([2, 3], 0xFF00);
        let b = Tensor::<u64, Dim2>::full([2, 3], 0x0FF0);
        let c = a.bitor(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_bitnot_u64() {
        let a = Tensor::<u64, Dim2>::full([2, 3], 0xFF);
        let c = a.bitnot();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_shl_u64() {
        let a = Tensor::<u64, Dim2>::full([2, 3], 1);
        let b = Tensor::<u64, Dim2>::full([2, 3], 8);
        let c = a.shl(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    // Test with i8 type
    #[test]
    fn test_bitand_i8() {
        let a = Tensor::<i8, Dim2>::full([2, 3], 0b1111);
        let b = Tensor::<i8, Dim2>::full([2, 3], 0b0101);
        let c = a.bitand(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }
}
