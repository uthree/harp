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

use crate::ast::DType;

// ============================================================================
// Macros for impl generation
// ============================================================================

/// Macro to implement integer type base traits (TensorDType, NumericDType)
/// IntegerDType and its subtypes are implemented in mod.rs with extended functionality
macro_rules! impl_integer_base {
    ($($ty:ty => $dtype:expr);+ $(;)?) => {
        $(
            impl TensorDType for $ty {
                const DTYPE: DType = $dtype;
            }
            impl NumericDType for $ty {}
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

/// Marker trait for numeric types (integers and floats).
///
/// Types implementing this trait support basic arithmetic operations
/// (Add, Sub, Mul, Div).
pub trait NumericDType: TensorDType {}

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
impl NumericDType for f32 {}
impl NumericDType for f64 {}
// FloatDType implementations are in mod.rs

// Integer types (via macro) - IntegerDType impl is in mod.rs
impl_integer_base!(
    i8  => DType::I8;
    i16 => DType::I16;
    i32 => DType::I32;
    i64 => DType::I64;
    u8  => DType::U8;
    u16 => DType::U16;
    u32 => DType::U32;
    u64 => DType::U64;
);

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
    }
}
