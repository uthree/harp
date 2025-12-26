//! Tensor data type traits
//!
//! This module defines the trait hierarchy for tensor element types:
//!
//! - `TensorDType`: Base trait for all tensor element types
//! - `NumericDType`: Types that support arithmetic operations (integers and floats)
//! - `FloatDType`: Floating-point types (f32, f64) - support transcendental functions and gradients
//! - `IntegerDType`: Integer types - support bitwise operations
//! - `SignedIntDType`: Signed integer types (i8, i16, i32, i64)
//! - `UnsignedIntDType`: Unsigned integer types (u8, u16, u32)

use crate::ast::DType;

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

/// Marker trait for numeric types (integers and floats).
///
/// Types implementing this trait support basic arithmetic operations
/// (Add, Sub, Mul, Div).
pub trait NumericDType: TensorDType {}

/// Marker trait for floating-point types (f32, f64).
///
/// Types implementing this trait support:
/// - Transcendental functions (sin, cos, exp, log, etc.)
/// - Gradient computation (autograd)
/// - Floor, ceil, round operations
pub trait FloatDType: NumericDType {}

/// Marker trait for integer types (signed and unsigned).
///
/// Types implementing this trait support:
/// - Bitwise operations (and, or, xor, shl, shr)
pub trait IntegerDType: NumericDType {}

/// Marker trait for signed integer types (i8, i16, i32, i64).
pub trait SignedIntDType: IntegerDType {}

/// Marker trait for unsigned integer types (u8, u16, u32).
pub trait UnsignedIntDType: IntegerDType {}

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
impl NumericDType for f32 {}
impl NumericDType for f64 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}

// Signed integer types
impl TensorDType for i8 {
    const DTYPE: DType = DType::I8;
}
impl TensorDType for i16 {
    const DTYPE: DType = DType::I16;
}
impl TensorDType for i32 {
    const DTYPE: DType = DType::I32;
}
impl TensorDType for i64 {
    const DTYPE: DType = DType::I64;
}
impl NumericDType for i8 {}
impl NumericDType for i16 {}
impl NumericDType for i32 {}
impl NumericDType for i64 {}
impl IntegerDType for i8 {}
impl IntegerDType for i16 {}
impl IntegerDType for i32 {}
impl IntegerDType for i64 {}
impl SignedIntDType for i8 {}
impl SignedIntDType for i16 {}
impl SignedIntDType for i32 {}
impl SignedIntDType for i64 {}

// Unsigned integer types
impl TensorDType for u8 {
    const DTYPE: DType = DType::U8;
}
impl TensorDType for u16 {
    const DTYPE: DType = DType::U16;
}
impl TensorDType for u32 {
    const DTYPE: DType = DType::U32;
}
impl NumericDType for u8 {}
impl NumericDType for u16 {}
impl NumericDType for u32 {}
impl IntegerDType for u8 {}
impl IntegerDType for u16 {}
impl IntegerDType for u32 {}
impl UnsignedIntDType for u8 {}
impl UnsignedIntDType for u16 {}
impl UnsignedIntDType for u32 {}

// Boolean type (not numeric)
impl TensorDType for bool {
    const DTYPE: DType = DType::Bool;
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
    }

    #[test]
    fn test_tensor_dtype_bool() {
        assert_eq!(bool::DTYPE, DType::Bool);
    }

    // Type-level tests: ensure trait bounds work as expected
    fn requires_numeric<T: NumericDType>() {}
    fn requires_float<T: FloatDType>() {}
    fn requires_integer<T: IntegerDType>() {}
    fn requires_signed<T: SignedIntDType>() {}
    fn requires_unsigned<T: UnsignedIntDType>() {}

    #[test]
    fn test_trait_bounds() {
        // Float types satisfy NumericDType and FloatDType
        requires_numeric::<f32>();
        requires_numeric::<f64>();
        requires_float::<f32>();
        requires_float::<f64>();

        // Signed integers satisfy NumericDType, IntegerDType, SignedIntDType
        requires_numeric::<i32>();
        requires_integer::<i32>();
        requires_signed::<i32>();

        // Unsigned integers satisfy NumericDType, IntegerDType, UnsignedIntDType
        requires_numeric::<u32>();
        requires_integer::<u32>();
        requires_unsigned::<u32>();
    }
}
