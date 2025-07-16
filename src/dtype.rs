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
