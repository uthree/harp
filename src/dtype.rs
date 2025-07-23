use std::fmt::{Debug, Display, Formatter, Result};

pub trait IsNumber: Clone + Copy + Debug + Display {
    fn dtype() -> DType;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Unit,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::Unit => 0,
        }
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            DType::F32 => write!(f, "float"),
            DType::F64 => write!(f, "double"),
            DType::I8 => write!(f, "char"),
            DType::I16 => write!(f, "short"),
            DType::I32 => write!(f, "int"),
            DType::I64 => write!(f, "long"),
            DType::U8 => write!(f, "unsigned char"),
            DType::U16 => write!(f, "unsigned short"),
            DType::U32 => write!(f, "unsigned int"),
            DType::U64 => write!(f, "unsigned long"),
            DType::Unit => write!(f, "void"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Number {
    F32(f32),
    F64(f64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

macro_rules! impl_is_number {
    ($($t:ty, $dtype:ident),*) => {
        $(
            impl IsNumber for $t {
                fn dtype() -> DType {
                    DType::$dtype
                }
            }
        )*
    };
}

impl_is_number! {
    f32, F32,
    f64, F64,
    i8, I8,
    i16, I16,
    i32, I32,
    i64, I64,
    u8, U8,
    u16, U16,
    u32, U32,
    u64, U64
}

macro_rules! impl_from_for_number {
    ($($t:ty, $variant:ident),*) => {
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
    f32, F32,
    f64, F64,
    i8, I8,
    i16, I16,
    i32, I32,
    i64, I64,
    u8, U8,
    u16, U16,
    u32, U32,
    u64, U64
}

impl Display for Number {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Number::F32(n) => write!(f, "{n}f"),
            Number::F64(n) => write!(f, "{n}"),
            Number::I8(n) => write!(f, "{n}"),
            Number::I16(n) => write!(f, "{n}"),
            Number::I32(n) => write!(f, "{n}"),
            Number::I64(n) => write!(f, "{n}L"),
            Number::U8(n) => write!(f, "{n}"),
            Number::U16(n) => write!(f, "{n}"),
            Number::U32(n) => write!(f, "{n}U"),
            Number::U64(n) => write!(f, "{n}UL"),
        }
    }
}

