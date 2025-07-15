use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};
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

// operator types
#[derive(Clone, PartialEq, Debug)]
pub enum Ops {
    Add,
    Mul,
    Recip,
    Rem,
    Cast(DType),
    Const(Number),
    Exp2,
    Log2,
    Sin,
    Sqrt,
}

// internal data of UOp
#[derive(Clone, PartialEq, Debug)]
struct UOp_ {
    op: Ops,
    dtype: DType,
    src: Vec<UOp>,
}

// micro operator
#[derive(Clone, PartialEq, Debug)]
pub struct UOp(Rc<UOp_>);

impl UOp {
    pub fn new(op: Ops, dtype: DType, src: Vec<UOp>) -> Self {
        UOp(Rc::new(UOp_ { op, dtype, src }))
    }

    pub fn recip(self) -> Self {
        // TODO: Implement proper dtype promotion
        let dtype = self.0.dtype.clone();
        UOp::new(Ops::Recip, dtype, vec![self])
    }

    pub fn cast(self, dtype: DType) -> Self {
        UOp::new(Ops::Cast(dtype.clone()), dtype, vec![self])
    }

    pub fn exp2(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Ops::Exp2, dtype, vec![self])
    }

    pub fn log2(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Ops::Log2, dtype, vec![self])
    }

    pub fn sin(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Ops::Sin, dtype, vec![self])
    }

    pub fn sqrt(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Ops::Sqrt, dtype, vec![self])
    }
}

macro_rules! impl_from_for_uop {
    ($($t:ty => ($variant:ident, $dtype:ident)),*) => {
        $(
            impl From<$t> for UOp {
                fn from(n: $t) -> Self {
                    UOp::new(
                        Ops::Const(Number::$variant(n)),
                        DType::$dtype,
                        vec![],
                    )
                }
            }
        )*
    };
}

impl_from_for_uop! {
    u8 => (U8, U8), u16 => (U16, U16), u32 => (U32, U32), u64 => (U64, U64),
    i8 => (I8, I8), i16 => (I16, I16), i32 => (I32, I32), i64 => (I64, I64),
    f32 => (F32, F32), f64 => (F64, F64)
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:ident) => {
        impl $trait for UOp {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                // TODO: Implement proper dtype promotion
                let dtype = self.0.dtype.clone();
                UOp::new(Ops::$op, dtype, vec![self, rhs])
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Mul, mul, Mul);
impl_binary_op!(Rem, rem, Rem);

impl Sub for UOp {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let neg_one: UOp = match &self.0.dtype {
            DType::I8 => (-1i8).into(),
            DType::I16 => (-1i16).into(),
            DType::I32 => (-1i32).into(),
            DType::I64 => (-1i64).into(),
            DType::F32 => (-1.0f32).into(),
            DType::F64 => (-1.0f64).into(),
            dtype => unimplemented!("Subtraction is not implemented for dtype {:?}", dtype),
        };
        self + (rhs * neg_one)
    }
}

impl Div for UOp {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op_trait:ident, $op_method:ident) => {
        impl $trait for UOp {
            fn $method(&mut self, rhs: Self) {
                *self = self.clone().$op_method(rhs);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, Add, add);
impl_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_assign_op!(DivAssign, div_assign, Div, div);
impl_assign_op!(RemAssign, rem_assign, Rem, rem);

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
    fn test_uop_from_numeric() {
        let uop_from_i32: UOp = 5i32.into();
        assert_eq!(uop_from_i32.0.op, Ops::Const(Number::I32(5)));
        assert_eq!(uop_from_i32.0.dtype, DType::I32);
        assert!(uop_from_i32.0.src.is_empty());

        let uop_from_f64: UOp = 3.14f64.into();
        assert_eq!(uop_from_f64.0.op, Ops::Const(Number::F64(3.14)));
        assert_eq!(uop_from_f64.0.dtype, DType::F64);

        let uop_from_u8: UOp = 255u8.into();
        assert_eq!(uop_from_u8.0.op, Ops::Const(Number::U8(255)));
        assert_eq!(uop_from_u8.0.dtype, DType::U8);
    }

    #[test]
    fn test_binary_operations() {
        let a: UOp = 5i32.into();
        let b: UOp = 10i32.into();

        // Test Add
        let c = a.clone() + b.clone();
        assert_eq!(c.0.op, Ops::Add);
        assert_eq!(c.0.dtype, DType::I32);
        assert_eq!(c.0.src.len(), 2);
        assert_eq!(c.0.src[0], a);
        assert_eq!(c.0.src[1], b);

        // Test Mul
        let d = a.clone() * b.clone();
        assert_eq!(d.0.op, Ops::Mul);
        assert_eq!(d.0.dtype, DType::I32);
        assert_eq!(d.0.src.len(), 2);
        assert_eq!(d.0.src[0], a);
        assert_eq!(d.0.src[1], b);

        // Test Sub
        let e = a.clone() - b.clone();
        assert_eq!(e.0.op, Ops::Add); // a + (b * -1)
        assert_eq!(e.0.src.len(), 2);
        assert_eq!(e.0.src[0], a.clone());
        let sub_rhs = e.0.src[1].clone();
        assert_eq!(sub_rhs.0.op, Ops::Mul); // b * -1
        assert_eq!(sub_rhs.0.src.len(), 2);
        assert_eq!(sub_rhs.0.src[0], b.clone());
        let neg_one = sub_rhs.0.src[1].clone();
        assert_eq!(neg_one.0.op, Ops::Const(Number::I32(-1)));

        // Test Div
        let f = a.clone() / b.clone();
        assert_eq!(f.0.op, Ops::Mul); // a * b.recip()
        assert_eq!(f.0.src.len(), 2);
        assert_eq!(f.0.src[0], a.clone());
        let div_rhs = f.0.src[1].clone();
        assert_eq!(div_rhs.0.op, Ops::Recip);
        assert_eq!(div_rhs.0.src.len(), 1);
        assert_eq!(div_rhs.0.src[0], b.clone());

        // Test Rem
        let g = a.clone() % b.clone();
        assert_eq!(g.0.op, Ops::Rem);
        assert_eq!(g.0.dtype, DType::I32);
        assert_eq!(g.0.src.len(), 2);
        assert_eq!(g.0.src[0], a.clone());
        assert_eq!(g.0.src[1], b.clone());

        // Test Sub with f64
        let x: UOp = 5.0f64.into();
        let y: UOp = 10.0f64.into();
        let z = x.clone() - y.clone();
        assert_eq!(z.0.op, Ops::Add);
        let sub_rhs_f64 = z.0.src[1].clone();
        assert_eq!(sub_rhs_f64.0.op, Ops::Mul);
        let neg_one_f64 = sub_rhs_f64.0.src[1].clone();
        assert_eq!(neg_one_f64.0.op, Ops::Const(Number::F64(-1.0)));
    }

    #[test]
    fn test_assign_operations() {
        let b: UOp = 10i32.into();

        let mut a: UOp = 5i32.into();
        a += b.clone();
        let expected: UOp = 5i32.into();
        assert_eq!(a, expected + b.clone());

        let mut a: UOp = 5i32.into();
        a -= b.clone();
        let expected: UOp = 5i32.into();
        assert_eq!(a, expected - b.clone());

        let mut a: UOp = 5i32.into();
        a *= b.clone();
        let expected: UOp = 5i32.into();
        assert_eq!(a, expected * b.clone());

        let mut a: UOp = 5i32.into();
        a /= b.clone();
        let expected: UOp = 5i32.into();
        assert_eq!(a, expected / b.clone());

        let mut a: UOp = 5i32.into();
        a %= b.clone();
        let expected: UOp = 5i32.into();
        assert_eq!(a, expected % b.clone());
    }

    #[test]
    fn test_cast_operation() {
        let a: UOp = 5i32.into();
        let b = a.clone().cast(DType::F64);

        assert_eq!(b.0.op, Ops::Cast(DType::F64));
        assert_eq!(b.0.dtype, DType::F64);
        assert_eq!(b.0.src.len(), 1);
        assert_eq!(b.0.src[0], a);
    }

    #[test]
    fn test_number_dtype() {
        let num_i32 = Number::I32(42);
        assert_eq!(num_i32.dtype(), DType::I32);

        let num_f64 = Number::F64(3.14);
        assert_eq!(num_f64.dtype(), DType::F64);

        let num_u8 = Number::U8(255);
        assert_eq!(num_u8.dtype(), DType::U8);
    }

    #[test]
    fn test_unary_operations() {
        let a: UOp = 5i32.into();

        // Test Exp2
        let b = a.clone().exp2();
        assert_eq!(b.0.op, Ops::Exp2);
        assert_eq!(b.0.dtype, DType::I32);
        assert_eq!(b.0.src.len(), 1);
        assert_eq!(b.0.src[0], a);

        // Test Log2
        let c = a.clone().log2();
        assert_eq!(c.0.op, Ops::Log2);
        assert_eq!(c.0.dtype, DType::I32);
        assert_eq!(c.0.src.len(), 1);
        assert_eq!(c.0.src[0], a);

        // Test Sin
        let d = a.clone().sin();
        assert_eq!(d.0.op, Ops::Sin);
        assert_eq!(d.0.dtype, DType::I32);
        assert_eq!(d.0.src.len(), 1);
        assert_eq!(d.0.src[0], a);

        // Test Sqrt
        let e = a.clone().sqrt();
        assert_eq!(e.0.op, Ops::Sqrt);
        assert_eq!(e.0.dtype, DType::I32);
        assert_eq!(e.0.src.len(), 1);
        assert_eq!(e.0.src[0], a);
    }
}