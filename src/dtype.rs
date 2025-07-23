use std::fmt;

// datatypes
#[derive(Clone, PartialEq, Debug)]
pub enum DType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Pointer(Box<DType>, usize),
    Unit,
}

impl DType {
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
            Number::F32(n) => write!(f, "{n}f"),
            Number::F64(n) => write!(f, "{n}"),
        }
    }
}

macro_rules! impl_number_dtype {
    ($($variant:ident),*) => {
        impl Number {
            pub fn dtype(&self) -> DType {
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
