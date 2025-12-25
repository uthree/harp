//! Data type definitions for tensors
//!
//! Defines the supported data types for tensor operations.

/// Tensor data type
///
/// Unlike AST's DType, this does not include Vec or Ptr types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    /// Unknown or placeholder type (for type inference)
    Unknown,
    /// Boolean type (internally u8: 0 = false, non-zero = true)
    Bool,
    /// 64-bit signed integer (for indexing/counters)
    I64,
    /// 32-bit signed integer
    I32,
    /// 32-bit floating point
    F32,
}

impl Default for DType {
    fn default() -> Self {
        DType::Unknown
    }
}

impl DType {
    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32)
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self, DType::I32 | DType::I64)
    }

    /// Check if this is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Check if the type is known (not Unknown)
    pub fn is_known(&self) -> bool {
        !matches!(self, DType::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_is_float() {
        assert!(DType::F32.is_float());
        assert!(!DType::I32.is_float());
    }

    #[test]
    fn test_dtype_is_integer() {
        assert!(DType::I32.is_integer());
        assert!(DType::I64.is_integer());
        assert!(!DType::F32.is_integer());
    }

    #[test]
    fn test_dtype_default() {
        assert_eq!(DType::default(), DType::Unknown);
    }
}
