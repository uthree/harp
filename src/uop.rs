use crate::dtype::DType;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
enum Ops {
    Const(ConstantNumber),

    // mathematical operators
    Add(UOp, UOp),
    Mul(UOp, UOp),
    Rem(UOp, UOp),
    Recip(UOp),
    Sqrt(UOp),
    Sin(UOp),
    Log2(UOp),
    Exp2(UOp),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UOp {
    rc: Rc<Ops>,
    dtype: DType,
}

#[derive(Clone, Debug, PartialEq)]
enum ConstantNumber {
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

macro_rules! impl_from_for_uop {
    ($from_type:ty, $variant:ident) => {
        impl From<$from_type> for UOp {
            fn from(value: $from_type) -> Self {
                UOp {
                    rc: Rc::new(Ops::Const(ConstantNumber::$variant(value))),
                    dtype: DType::$variant,
                }
            }
        }
    };
}

impl_from_for_uop!(u8, U8);
impl_from_for_uop!(u16, U16);
impl_from_for_uop!(u32, U32);
impl_from_for_uop!(u64, U64);
impl_from_for_uop!(i8, I8);
impl_from_for_uop!(i16, I16);
impl_from_for_uop!(i32, I32);
impl_from_for_uop!(i64, I64);
impl_from_for_uop!(f32, F32);
impl_from_for_uop!(f64, F64);

// operator implementations

impl<T> std::ops::Add<T> for UOp
where
    T: Into<UOp>,
{
    type Output = UOp;
    fn add(self, rhs: T) -> Self::Output {
        UOp {
            rc: Rc::new(Ops::Add(self.clone(), rhs.into())),
            dtype: self.dtype,
        }
    }
}

impl<T> std::ops::Mul<T> for UOp
where
    T: Into<UOp>,
{
    type Output = UOp;
    fn mul(self, rhs: T) -> Self::Output {
        UOp {
            rc: Rc::new(Ops::Mul(self.clone(), rhs.into())),
            dtype: self.dtype,
        }
    }
}

impl<T> std::ops::Rem<T> for UOp
where
    T: Into<UOp>,
{
    type Output = UOp;
    fn rem(self, rhs: T) -> Self::Output {
        UOp {
            rc: Rc::new(Ops::Rem(self.clone(), rhs.into())),
            dtype: self.dtype,
        }
    }
}
