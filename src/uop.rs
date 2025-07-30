use std::{ops::Deref, rc::Rc};

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // placeholder for pattern matching
    Capture(usize),

    // Literal
    Const(Const),
    Var(String),

    // unary ops
    Neg,
    Recip,
    Sin,
    Sqrt,

    // binary ops
    Add,
    Mul,
    Max,
    Rem,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UOp_ {
    pub op: Op,
    pub src: Vec<UOp>,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UOp(Rc<UOp_>);

impl UOp {
    pub fn new(op: Op, src: Vec<UOp>, dtype: DType) -> Self {
        UOp(Rc::new(UOp_ { op, src, dtype }))
    }

    pub fn capture(id: usize) -> Self {
        UOp::new(Op::Capture(id), vec![], DType::Capture)
    }

    pub fn var(name: &str, dtype: DType) -> Self {
        UOp::new(Op::Var(name.to_string()), vec![], dtype)
    }
}

impl Deref for UOp {
    type Target = UOp_;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

macro_rules! impl_unary_op {
    ($op: ident, $fname: ident) => {
        impl UOp {
            fn $fname(self: Self) -> Self {
                UOp::new(Op::$op, vec![self.clone()], self.dtype.clone())
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl UOp {
            pub fn $fname(self: Self) -> Self {
                UOp::new(Op::$op, vec![self.clone()], self.dtype.clone())
            }
        }
    };
}

impl_unary_op!(Neg, neg_);
impl_unary_op!(Recip, recip);
impl_unary_op!(pub, Sqrt, sqrt);
impl_unary_op!(pub, Sin, sin);

macro_rules! impl_binary_op {
    ($op: ident, $fname: ident) => {
        impl UOp {
            fn $fname(self: Self, other: impl Into<UOp>) -> Self {
                let other = other.into();
                if self.dtype != DType::Capture && other.dtype != DType::Capture {
                    if self.dtype != other.dtype {
                        panic!(
                            "type mismatch: left: {:?}, right: {:?}",
                            self.dtype, other.dtype
                        );
                    }
                }
                UOp::new(Op::$op, vec![self.clone(), other], self.dtype.clone())
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl UOp {
            pub fn $fname(self: Self, other: impl Into<UOp>) -> Self {
                let other = other.into();
                if self.dtype != DType::Capture && other.dtype != DType::Capture {
                    if self.dtype != other.dtype {
                        panic!(
                            "type mismatch: left: {:?}, right: {:?}",
                            self.dtype, other.dtype
                        );
                    }
                }
                UOp::new(Op::$op, vec![self.clone(), other], self.dtype.clone())
            }
        }
    };
}

impl_binary_op!(Add, add_);
impl_binary_op!(Mul, mul_);
impl_binary_op!(pub, Max, max);
impl_binary_op!(Rem, rem_);

#[derive(Debug, Clone, PartialEq)]
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
    None, // void
    Capture,
    Ptr(Box<Self>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Const {
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

macro_rules! impl_dtype {
    ($variant: ident, $num_type: ident) => {
        impl From<$num_type> for Const {
            fn from(v: $num_type) -> Self {
                Const::$variant(v)
            }
        }

        impl From<$num_type> for UOp {
            fn from(v: $num_type) -> Self {
                UOp::new(Op::Const(Const::$variant(v)), vec![], DType::$variant)
            }
        }
    };
}

impl_dtype!(F32, f32);
impl_dtype!(F64, f64);
impl_dtype!(I8, i8);
impl_dtype!(I16, i16);
impl_dtype!(I32, i32);
impl_dtype!(I64, i64);
impl_dtype!(U8, u8);
impl_dtype!(U16, u16);
impl_dtype!(U32, u32);
impl_dtype!(U64, u64);

impl Const {
    pub fn dtype(&self) -> DType {
        match *self {
            Const::F32(_) => DType::F32,
            Const::F64(_) => DType::F64,
            Const::I8(_) => DType::I8,
            Const::I16(_) => DType::I16,
            Const::I32(_) => DType::I32,
            Const::I64(_) => DType::I64,
            Const::U8(_) => DType::U8,
            Const::U16(_) => DType::U16,
            Const::U32(_) => DType::U32,
            Const::U64(_) => DType::U64,
        }
    }
}

impl<T> std::ops::Add<T> for UOp
where
    T: Into<UOp>,
{
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        self.add_(rhs.into())
    }
}

impl<T> std::ops::Sub<T> for UOp
where
    T: Into<UOp>,
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        self.add_(rhs.into().neg_())
    }
}

impl<T> std::ops::Mul<T> for UOp
where
    T: Into<UOp>,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        self.mul_(rhs.into())
    }
}

impl<T> std::ops::Div<T> for UOp
where
    T: Into<UOp>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        self.mul_(rhs.into().recip())
    }
}

impl<T> std::ops::Rem<T> for UOp
where
    T: Into<UOp>,
{
    type Output = Self;
    fn rem(self, rhs: T) -> Self::Output {
        self.rem_(rhs.into())
    }
}

impl std::ops::Neg for UOp {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.neg_()
    }
}
