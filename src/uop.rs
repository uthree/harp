use std::rc::Rc;

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
    Pointer(Box<DType>),
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

// operator types
#[derive(Clone, PartialEq, Debug)]
pub enum Ops {
    Add,
    Mul,
    Const(Number),
}

macro_rules! impl_from_for_ops {
    ($($t:ty => $variant:ident),*) => {
        $(
            impl From<$t> for Ops {
                fn from(n: $t) -> Self {
                    Ops::Const(Number::$variant(n))
                }
            }
        )*
    };
}

impl_from_for_ops! {
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    f32 => F32, f64 => F64
}

// internal data of UOp
#[derive(Clone, PartialEq, Debug)]
struct UOp_ {
    op: DType,
    src: Vec<UOp>,
}

// micro operator
#[derive(Clone, PartialEq, Debug)]
pub struct UOp(Rc<UOp_>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_creation_and_comparison() {
        let u8_type = DType::U8;
        let u16_type = DType::U16;
        let u32_type = DType::U32;
        let u64_type = DType::U64;
        let i8_type = DType::I8;
        let i16_type = DType::I16;
        let i32_type = DType::I32;
        let i64_type = DType::I64;
        let f32_type = DType::F32;
        let f64_type = DType::F64;

        assert_eq!(u8_type, DType::U8);
        assert_ne!(u8_type, u16_type);
        assert_eq!(u16_type, DType::U16);
        assert_eq!(u32_type, DType::U32);
        assert_eq!(u64_type, DType::U64);
        assert_eq!(i8_type, DType::I8);
        assert_eq!(i16_type, DType::I16);
        assert_eq!(i32_type, DType::I32);
        assert_eq!(i64_type, DType::I64);
        assert_eq!(f32_type, DType::F32);
        assert_eq!(f64_type, DType::F64);
    }

    #[test]
    fn test_ops_from_numeric() {
        let ops_from_i32: Ops = 5i32.into();
        assert_eq!(ops_from_i32, Ops::Const(Number::I32(5)));

        let ops_from_f64: Ops = 3.14f64.into();
        assert_eq!(ops_from_f64, Ops::Const(Number::F64(3.14)));

        let ops_from_u8: Ops = 255u8.into();
        assert_eq!(ops_from_u8, Ops::Const(Number::U8(255)));
    }
}
