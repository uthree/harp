//! Data types for tensor elements.

use std::fmt;

/// Supported data types for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    Bool,
    Int32,
    Int64,
    #[default]
    Float32,
    Float64,
}

impl DType {
    /// Returns the size of this type in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Float32 => 4,
            DType::Float64 => 8,
        }
    }

    /// Returns true if this is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    /// Returns true if this is an integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, DType::Int32 | DType::Int64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::Int32 => write!(f, "i32"),
            DType::Int64 => write!(f, "i64"),
            DType::Float32 => write!(f, "f32"),
            DType::Float64 => write!(f, "f64"),
        }
    }
}

/// Trait for scalar types that can be stored in tensors.
pub trait Scalar: Copy + Send + Sync + 'static {
    const DTYPE: DType;

    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn to_bytes(self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl Scalar for bool {
    const DTYPE: DType = DType::Bool;

    fn to_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }

    fn from_f64(v: f64) -> Self {
        v != 0.0
    }

    fn to_bytes(self) -> Vec<u8> {
        vec![if self { 1 } else { 0 }]
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

impl Scalar for i32 {
    const DTYPE: DType = DType::Int32;

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(v: f64) -> Self {
        v as i32
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        i32::from_ne_bytes(bytes[..4].try_into().unwrap())
    }
}

impl Scalar for i64 {
    const DTYPE: DType = DType::Int64;

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(v: f64) -> Self {
        v as i64
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        i64::from_ne_bytes(bytes[..8].try_into().unwrap())
    }
}

impl Scalar for f32 {
    const DTYPE: DType = DType::Float32;

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        f32::from_ne_bytes(bytes[..4].try_into().unwrap())
    }
}

impl Scalar for f64 {
    const DTYPE: DType = DType::Float64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(v: f64) -> Self {
        v
    }

    fn to_bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        f64::from_ne_bytes(bytes[..8].try_into().unwrap())
    }
}

/// A type-erased scalar value.
#[derive(Debug, Clone, Copy)]
pub enum ScalarValue {
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
}

impl ScalarValue {
    pub fn dtype(&self) -> DType {
        match self {
            ScalarValue::Bool(_) => DType::Bool,
            ScalarValue::Int32(_) => DType::Int32,
            ScalarValue::Int64(_) => DType::Int64,
            ScalarValue::Float32(_) => DType::Float32,
            ScalarValue::Float64(_) => DType::Float64,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            ScalarValue::Bool(v) => v.to_f64(),
            ScalarValue::Int32(v) => v.to_f64(),
            ScalarValue::Int64(v) => v.to_f64(),
            ScalarValue::Float32(v) => v.to_f64(),
            ScalarValue::Float64(v) => v.to_f64(),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            ScalarValue::Bool(v) => v.to_bytes(),
            ScalarValue::Int32(v) => v.to_bytes(),
            ScalarValue::Int64(v) => v.to_bytes(),
            ScalarValue::Float32(v) => v.to_bytes(),
            ScalarValue::Float64(v) => v.to_bytes(),
        }
    }

    pub fn cast(&self, dtype: DType) -> ScalarValue {
        let f = self.to_f64();
        match dtype {
            DType::Bool => ScalarValue::Bool(bool::from_f64(f)),
            DType::Int32 => ScalarValue::Int32(i32::from_f64(f)),
            DType::Int64 => ScalarValue::Int64(i64::from_f64(f)),
            DType::Float32 => ScalarValue::Float32(f32::from_f64(f)),
            DType::Float64 => ScalarValue::Float64(f64::from_f64(f)),
        }
    }
}

impl<T: Scalar> From<T> for ScalarValue {
    fn from(v: T) -> Self {
        match T::DTYPE {
            DType::Bool => ScalarValue::Bool(bool::from_f64(v.to_f64())),
            DType::Int32 => ScalarValue::Int32(i32::from_f64(v.to_f64())),
            DType::Int64 => ScalarValue::Int64(i64::from_f64(v.to_f64())),
            DType::Float32 => ScalarValue::Float32(f32::from_f64(v.to_f64())),
            DType::Float64 => ScalarValue::Float64(f64::from_f64(v.to_f64())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Bool.size_bytes(), 1);
        assert_eq!(DType::Int32.size_bytes(), 4);
        assert_eq!(DType::Int64.size_bytes(), 8);
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert_eq!(DType::Float64.size_bytes(), 8);
    }

    #[test]
    fn test_dtype_is_float() {
        assert!(!DType::Bool.is_float());
        assert!(!DType::Int32.is_float());
        assert!(DType::Float32.is_float());
        assert!(DType::Float64.is_float());
    }

    #[test]
    fn test_scalar_roundtrip() {
        let v: f32 = 3.14;
        let bytes = v.to_bytes();
        let v2 = f32::from_bytes(&bytes);
        assert_eq!(v, v2);
    }
}
