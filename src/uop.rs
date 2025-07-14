use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
enum Ops {
    Const(ConstantNumber),
    Add(UOp, UOp),
    Mul(UOp, UOp),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UOp(Rc<Ops>);

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
    Usize(usize),
    Isize(isize),
    F32(f32),
    F64(f64),
}

macro_rules! impl_from_for_uop {
    ($from_type:ty, $variant:ident) => {
        impl From<$from_type> for UOp {
            fn from(value: $from_type) -> Self {
                UOp(Rc::new(Ops::Const(ConstantNumber::$variant(value))))
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
impl_from_for_uop!(usize, Usize);
impl_from_for_uop!(isize, Isize);
impl_from_for_uop!(f32, F32);
impl_from_for_uop!(f64, F64);

impl std::ops::Add for UOp {
    type Output = UOp;
    fn add(self, rhs: Self) -> Self::Output {
        UOp(Rc::new(Ops::Add(self, rhs)))
    }
}

impl std::ops::Mul for UOp {
    type Output = UOp;
    fn mul(self, rhs: Self) -> Self::Output {
        UOp(Rc::new(Ops::Mul(self, rhs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_from_for_uop {
        ($test_name:ident, $from_type:ty, $variant:ident, $value:expr) => {
            #[test]
            fn $test_name() {
                let uop: UOp = ($value as $from_type).into();
                assert_eq!(
                    uop,
                    UOp(Rc::new(Ops::Const(ConstantNumber::$variant(
                        $value as $from_type
                    ))))
                );
            }
        };
    }

    test_from_for_uop!(test_from_u8, u8, U8, 8);
    test_from_for_uop!(test_from_u16, u16, U16, 16);
    test_from_for_uop!(test_from_u32, u32, U32, 32);
    test_from_for_uop!(test_from_u64, u64, U64, 64);
    test_from_for_uop!(test_from_i8, i8, I8, -8);
    test_from_for_uop!(test_from_i16, i16, I16, -16);
    test_from_for_uop!(test_from_i32, i32, I32, -32);
    test_from_for_uop!(test_from_i64, i64, I64, -64);
    test_from_for_uop!(test_from_usize, usize, Usize, 128);
    test_from_for_uop!(test_from_isize, isize, Isize, -128);
    test_from_for_uop!(test_from_f32, f32, F32, 3.2);
    test_from_for_uop!(test_from_f64, f64, F64, 6.4);
}
