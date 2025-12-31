//! Tensor data type traits
//!
//! This module defines the trait hierarchy for tensor element types:
//!
//! - `TensorDType`: Base trait for all tensor element types
//! - `NumericDType`: Types that support arithmetic operations (integers and floats)
//! - `FloatDType`: Floating-point types (f32, f64) - support transcendental functions and gradients
//! - `IntegerDType`: Integer types - support bitwise operations
//! - `SignedIntDType`: Signed integer types (i8, i16, i32, i64)
//! - `UnsignedIntDType`: Unsigned integer types (u8, u16, u32, u64)

use crate::ast::{DType, Literal};

// ============================================================================
// Macros for impl generation
// ============================================================================

/// Macro to implement integer type base traits (TensorDType, NumericDType)
/// IntegerDType and its subtypes are implemented in mod.rs with extended functionality
macro_rules! impl_integer_base {
    ($($ty:ty => $dtype:expr, $literal:ident);+ $(;)?) => {
        $(
            impl TensorDType for $ty {
                const DTYPE: DType = $dtype;
            }
            impl NumericDType for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
                fn to_literal(val: Self) -> Literal {
                    Literal::$literal(val)
                }
            }
        )+
    };
}

// ============================================================================
// TensorDType trait hierarchy
// ============================================================================

/// Trait for types that can be used as tensor element types.
///
/// This is the base trait for all tensor data types. Specific operations
/// require more constrained subtypes (e.g., `FloatDType` for trigonometric functions).
pub trait TensorDType: Clone + Send + Sync + 'static {
    /// The corresponding DType enum variant
    const DTYPE: DType;
}

/// Trait for numeric types (integers and floats).
///
/// Types implementing this trait support basic arithmetic operations
/// (Add, Sub, Mul, Div) and can be used for tensor initialization.
pub trait NumericDType: TensorDType {
    /// Zero value for this type
    const ZERO: Self;
    /// One value for this type
    const ONE: Self;
    /// Convert value to Literal for ConstFill operations
    fn to_literal(val: Self) -> Literal;
}

// FloatDType is defined in mod.rs with autograd methods
// IntegerDType, SignedIntDType, UnsignedIntDType are defined in mod.rs with extended functionality

// ============================================================================
// TensorDType implementations
// ============================================================================

// Floating-point types
impl TensorDType for f32 {
    const DTYPE: DType = DType::F32;
}
impl TensorDType for f64 {
    const DTYPE: DType = DType::F64;
}
impl NumericDType for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn to_literal(val: Self) -> Literal {
        Literal::F32(val)
    }
}
impl NumericDType for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn to_literal(val: Self) -> Literal {
        Literal::F64(val)
    }
}
// FloatDType implementations are in mod.rs

// Integer types (via macro) - IntegerDType impl is in mod.rs
impl_integer_base!(
    i8  => DType::I8,  I8;
    i16 => DType::I16, I16;
    i32 => DType::I32, I32;
    i64 => DType::I64, I64;
    u8  => DType::U8,  U8;
    u16 => DType::U16, U16;
    u32 => DType::U32, U32;
    u64 => DType::U64, U64;
);

// Boolean type (not numeric)
impl TensorDType for bool {
    const DTYPE: DType = DType::Bool;
}

// Complex types
use super::complex::Complex;

impl TensorDType for Complex<f32> {
    const DTYPE: DType = DType::Complex32;
}

impl TensorDType for Complex<f64> {
    const DTYPE: DType = DType::Complex64;
}

impl NumericDType for Complex<f32> {
    const ZERO: Self = Complex { re: 0.0, im: 0.0 };
    const ONE: Self = Complex { re: 1.0, im: 0.0 };

    fn to_literal(val: Self) -> Literal {
        Literal::Complex32(val.re, val.im)
    }
}

impl NumericDType for Complex<f64> {
    const ZERO: Self = Complex { re: 0.0, im: 0.0 };
    const ONE: Self = Complex { re: 1.0, im: 0.0 };

    fn to_literal(val: Self) -> Literal {
        Literal::Complex64(val.re, val.im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dtype_f32() {
        assert_eq!(f32::DTYPE, DType::F32);
    }

    #[test]
    fn test_tensor_dtype_f64() {
        assert_eq!(f64::DTYPE, DType::F64);
    }

    #[test]
    fn test_tensor_dtype_integers() {
        assert_eq!(i8::DTYPE, DType::I8);
        assert_eq!(i16::DTYPE, DType::I16);
        assert_eq!(i32::DTYPE, DType::I32);
        assert_eq!(i64::DTYPE, DType::I64);
        assert_eq!(u8::DTYPE, DType::U8);
        assert_eq!(u16::DTYPE, DType::U16);
        assert_eq!(u32::DTYPE, DType::U32);
        assert_eq!(u64::DTYPE, DType::U64);
    }

    #[test]
    fn test_tensor_dtype_bool() {
        assert_eq!(bool::DTYPE, DType::Bool);
    }

    // Type-level tests: ensure trait bounds work as expected
    fn requires_numeric<T: NumericDType>() {}

    #[test]
    fn test_trait_bounds() {
        // Float types satisfy NumericDType (FloatDType is tested in mod.rs)
        requires_numeric::<f32>();
        requires_numeric::<f64>();

        // Integer types satisfy NumericDType (IntegerDType is tested in mod.rs)
        requires_numeric::<i8>();
        requires_numeric::<i16>();
        requires_numeric::<i32>();
        requires_numeric::<i64>();
        requires_numeric::<u8>();
        requires_numeric::<u16>();
        requires_numeric::<u32>();
        requires_numeric::<u64>();

        // Complex types satisfy NumericDType
        requires_numeric::<Complex<f32>>();
        requires_numeric::<Complex<f64>>();
    }

    #[test]
    fn test_tensor_dtype_complex() {
        assert_eq!(Complex::<f32>::DTYPE, DType::Complex32);
        assert_eq!(Complex::<f64>::DTYPE, DType::Complex64);
    }

    #[test]
    fn test_numeric_dtype_complex() {
        // Test ZERO and ONE
        let zero_c32 = Complex::<f32>::ZERO;
        assert_eq!(zero_c32.re, 0.0);
        assert_eq!(zero_c32.im, 0.0);

        let one_c32 = Complex::<f32>::ONE;
        assert_eq!(one_c32.re, 1.0);
        assert_eq!(one_c32.im, 0.0);

        // Test to_literal
        let lit = NumericDType::to_literal(Complex::new(3.0f32, 4.0f32));
        assert!(matches!(lit, Literal::Complex32(3.0, 4.0)));

        let lit64 = NumericDType::to_literal(Complex::new(1.5f64, 2.5f64));
        assert!(matches!(lit64, Literal::Complex64(1.5, 2.5)));
    }
}
