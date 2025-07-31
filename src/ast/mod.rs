pub mod pattern;

use std::{boxed::Box, cell::Cell};

thread_local! {
    static NEXT_ID: Cell<usize> = Cell::new(0);
}

fn next_id() -> usize {
    NEXT_ID.with(|cell| {
        let id = cell.get();
        cell.set(id + 1);
        id
    })
}

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
    Log2,
    Exp2,

    // binary ops
    Add,
    Mul,
    Max,
    Rem,
}

#[derive(Debug, Clone)]
pub struct AstNode {
    pub id: usize,
    pub op: Op,
    pub src: Vec<Box<AstNode>>,
    pub dtype: DType,
}

impl PartialEq for AstNode {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.src == other.src && self.dtype == other.dtype
    }
}

impl AstNode {
    pub fn new(op: Op, src: Vec<Box<AstNode>>, dtype: DType) -> Self {
        Self {
            id: next_id(),
            op,
            src,
            dtype,
        }
    }

    pub fn capture(id: usize) -> Self {
        Self::new(Op::Capture(id), vec![], DType::Any)
    }

    pub fn var(name: &str, dtype: DType) -> Self {
        Self::new(Op::Var(name.to_string()), vec![], dtype)
    }

    pub fn with_type(self, dtype: DType) -> Self {
        Self::new(self.op, self.src, dtype)
    }
}

macro_rules! impl_unary_op {
    ($op: ident, $fname: ident) => {
        impl AstNode {
            fn $fname(self: Self) -> Self {
                let dtype = self.dtype.clone();
                AstNode::new(Op::$op, vec![Box::new(self)], dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self) -> Self {
                let dtype = self.dtype.clone();
                AstNode::new(Op::$op, vec![Box::new(self)], dtype)
            }
        }
    };
}

impl_unary_op!(Neg, neg_);
impl_unary_op!(Recip, recip);
impl_unary_op!(pub, Sqrt, sqrt);
impl_unary_op!(pub, Sin, sin);
impl_unary_op!(pub, Log2, log2);
impl_unary_op!(pub, Exp2, exp2);

macro_rules! impl_binary_op {
    ($op: ident, $fname: ident) => {
        impl AstNode {
            fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let other = other.into();
                let dtype = self.dtype.clone();
                if self.dtype != DType::Any && other.dtype != DType::Any {
                    if self.dtype != other.dtype {
                        panic!(
                            "type mismatch: left: {:?}, right: {:?}",
                            self.dtype, other.dtype
                        );
                    }
                }
                AstNode::new(Op::$op, vec![Box::new(self), Box::new(other)], dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let other = other.into();
                let dtype = self.dtype.clone();
                if self.dtype != DType::Any && other.dtype != DType::Any {
                    if self.dtype != other.dtype {
                        panic!(
                            "type mismatch: left: {:?}, right: {:?}",
                            self.dtype, other.dtype
                        );
                    }
                }
                AstNode::new(Op::$op, vec![Box::new(self), Box::new(other)], dtype)
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
    Any,  // Unchecked
    Ptr(Box<Self>, usize),
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

        impl From<$num_type> for AstNode {
            fn from(v: $num_type) -> Self {
                AstNode::new(Op::Const(Const::$variant(v)), vec![], DType::$variant)
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

impl<T> std::ops::Add<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        self.add_(rhs.into())
    }
}

impl<T> std::ops::Sub<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        self.add_(rhs.into().neg_())
    }
}

impl<T> std::ops::Mul<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        self.mul_(rhs.into())
    }
}

impl<T> std::ops::Div<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        self.mul_(rhs.into().recip())
    }
}

impl<T> std::ops::Rem<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn rem(self, rhs: T) -> Self::Output {
        self.rem_(rhs.into())
    }
}

impl std::ops::Neg for AstNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.neg_()
    }
}
