use std::any::TypeId;
use std::fmt;

/// Represents a data type in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    /// Returns the name of the data type.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Bool => "Bool",
            DType::I8 => "I8",
            DType::U8 => "U8",
            DType::I16 => "I16",
            DType::U16 => "U16",
            DType::I32 => "I32",
            DType::U32 => "U32",
            DType::I64 => "I64",
            DType::U64 => "U64",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::F32 => "F32",
            DType::F64 => "F64",
        }
    }

    /// Returns the `TypeId` of the corresponding Rust type.
    pub fn id(&self) -> TypeId {
        match self {
            DType::Bool => TypeId::of::<bool>(),
            DType::I8 => TypeId::of::<i8>(),
            DType::U8 => TypeId::of::<u8>(),
            DType::I16 => TypeId::of::<i16>(),
            DType::U16 => TypeId::of::<u16>(),
            DType::I32 => TypeId::of::<i32>(),
            DType::U32 => TypeId::of::<u32>(),
            DType::I64 => TypeId::of::<i64>(),
            DType::U64 => TypeId::of::<u64>(),
            // F16 and BF16 are not native types, so we can't get a TypeId for them directly.
            // This part might need adjustment based on how they are implemented.
            DType::F16 => TypeId::of::<f32>(), // Placeholder
            DType::BF16 => TypeId::of::<f32>(), // Placeholder
            DType::F32 => TypeId::of::<f32>(),
            DType::F64 => TypeId::of::<f64>(),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Represents a scalar value of any supported data type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Bool(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    // F16(f16),
    // BF16(bf16),
    F32(f32),
    F64(f64),
}

impl Scalar {
    /// Returns the `DType` of the scalar value.
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::I8(_) => DType::I8,
            Scalar::U8(_) => DType::U8,
            Scalar::I16(_) => DType::I16,
            Scalar::U16(_) => DType::U16,
            Scalar::I32(_) => DType::I32,
            Scalar::U32(_) => DType::U32,
            Scalar::I64(_) => DType::I64,
            Scalar::U64(_) => DType::U64,
            // Scalar::F16(_) => DType::F16,
            // Scalar::BF16(_) => DType::BF16,
            Scalar::F32(_) => DType::F32,
            Scalar::F64(_) => DType::F64,
        }
    }
}
