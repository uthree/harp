use std::fmt;

/// An enum representing the data types supported by the tensor library.
#[derive(Clone, PartialEq, Debug)]
pub enum DType {
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit floating-point.
    F32,
    /// 64-bit floating-point.
    F64,
    /// A pointer to another data type. The `usize` is for potential address space information.
    Pointer(Box<DType>, usize),
    /// A unit type, representing no value (similar to `void`).
    Unit,
}

impl DType {
    /// Returns the size of the data type in bytes.
    ///
    /// # Panics
    /// This function assumes a 64-bit architecture for pointer sizes.
    pub fn size(&self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
            DType::Pointer(_, _) => 8, // Assuming 64-bit pointers
            DType::Unit => 0,
        }
    }

    /// Returns the zero value for the data type.
    pub fn zero_value(&self) -> Number {
        match self {
            DType::F32 => Number::F32(0.0),
            DType::F64 => Number::F64(0.0),
            DType::I32 => Number::I32(0),
            DType::I64 => Number::I64(0),
            _ => unimplemented!("zero_value for {self:?} is not implemented"),
        }
    }

    /// Returns the one value for the data type.
    pub fn one_value(&self) -> Number {
        match self {
            DType::F32 => Number::F32(1.0),
            DType::F64 => Number::F64(1.0),
            DType::I32 => Number::I32(1),
            DType::I64 => Number::I64(1),
            _ => unimplemented!("one_value for {self:?} is not implemented"),
        }
    }

    /// Returns the minimum value for the data type.
    pub fn min_value(&self) -> Number {
        match self {
            DType::F32 => Number::F32(f32::MIN),
            DType::F64 => Number::F64(f64::MIN),
            DType::I32 => Number::I32(i32::MIN),
            DType::I64 => Number::I64(i64::MIN),
            _ => unimplemented!("min_value for {self:?} is not implemented"),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DType::U8 => write!(f, "unsigned char"),
            DType::U16 => write!(f, "unsigned short"),
            DType::U32 => write!(f, "unsigned int"),
            DType::U64 => write!(f, "unsigned long"),
            DType::I8 => write!(f, "char"),
            DType::I16 => write!(f, "short"),
            DType::I32 => write!(f, "int"),
            DType::I64 => write!(f, "long"),
            DType::F32 => write!(f, "float"),
            DType::F64 => write!(f, "double"),
            DType::Pointer(inner, _) => write!(f, "{inner}*"),
            DType::Unit => write!(f, "void"),
        }
    }
}

/// An enum to hold a numeric value of any supported data type.
///
/// This is primarily used for representing constant values within the `UOp` graph.
#[derive(Clone, PartialEq, Debug)]
pub enum Number {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Number::U8(n) => write!(f, "{n}"),
            Number::U16(n) => write!(f, "{n}"),
            Number::U32(n) => write!(f, "{n}"),
            Number::U64(n) => write!(f, "{n}"),
            Number::I8(n) => write!(f, "{n}"),
            Number::I16(n) => write!(f, "{n}"),
            Number::I32(n) => write!(f, "{n}"),
            Number::I64(n) => write!(f, "{n}"),
            Number::F32(n) => write!(f, "{n:e}f"),
            Number::F64(n) => write!(f, "{n:e}"),
        }
    }
}

macro_rules! impl_number_dtype {
    ($($variant:ident),*) => {
        impl Number {
            /// Returns the `DType` corresponding to the number's variant.
            pub fn get_dtype(&self) -> DType {
                match self {
                    $(
                        Number::$variant(_) => DType::$variant,
                    )*
                }
            }
        }
    };
}

impl_number_dtype!(U8, U16, U32, U64, I8, I16, I32, I64, F32, F64);

macro_rules! impl_from_for_number {
    ($($t:ty => $variant:ident),*) => {
        $(
            impl From<$t> for Number {
                fn from(n: $t) -> Self {
                    Number::$variant(n)
                }
            }
        )*
    };
}

impl_from_for_number! {
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    f32 => F32, f64 => F64
}
