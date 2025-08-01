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

#[derive(Debug, Clone, PartialEq)]
pub struct AstNode {
    pub id: usize,
    pub op: Op,
    pub src: Vec<Box<AstNode>>,
}

impl AstNode {
    pub fn new(op: Op, src: Vec<Box<AstNode>>) -> Self {
        Self {
            id: next_id(),
            op,
            src,
        }
    }

    pub fn capture(id: usize) -> Self {
        Self::new(Op::Capture(id), vec![])
    }

    pub fn var(name: &str) -> Self {
        Self::new(Op::Var(name.to_string()), vec![])
    }

    pub fn with_type(self) -> Self {
        Self::new(self.op, self.src)
    }
}

macro_rules! impl_unary_op {
    ($op: ident, $fname: ident) => {
        impl AstNode {
            fn $fname(self: Self) -> Self {
                AstNode::new(Op::$op, vec![Box::new(self)])
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self) -> Self {
                AstNode::new(Op::$op, vec![Box::new(self)])
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
                AstNode::new(Op::$op, vec![Box::new(self), Box::new(other)])
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let other = other.into();
                AstNode::new(Op::$op, vec![Box::new(self), Box::new(other)])
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
                AstNode::new(Op::Const(Const::$variant(v)), vec![])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, Op};
    #[test]
    fn test_unary_ops() {
        let a = AstNode::new(Op::Var("a".to_string()), vec![]);

        let neg_a = -a.clone();
        assert_eq!(neg_a.op, Op::Neg);
        assert_eq!(neg_a.src.len(), 1);
        assert_eq!(*neg_a.src[0], a);

        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let sqrt_a = a.clone().sqrt();
        assert_eq!(sqrt_a.op, Op::Sqrt);
        assert_eq!(sqrt_a.src.len(), 1);
        assert_eq!(*sqrt_a.src[0], a);

        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let sin_a = a.clone().sin();
        assert_eq!(sin_a.op, Op::Sin);
        assert_eq!(sin_a.src.len(), 1);
        assert_eq!(*sin_a.src[0], a);
    }

    #[test]
    fn test_binary_ops() {
        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let b = AstNode::new(Op::Var("b".to_string()), vec![]);

        let add_ab = a.clone() + b.clone();
        assert_eq!(add_ab.op, Op::Add);
        assert_eq!(add_ab.src.len(), 2);
        assert_eq!(*add_ab.src[0], a);
        assert_eq!(*add_ab.src[1], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let b = AstNode::new(Op::Var("b".to_string()), vec![]);
        let sub_ab = a.clone() - b.clone();
        assert_eq!(sub_ab.op, Op::Add); // sub is implemented as a + (-b)
        assert_eq!(sub_ab.src.len(), 2);
        assert_eq!(*sub_ab.src[0], a);
        assert_eq!(sub_ab.src[1].op, Op::Neg);
        assert_eq!(*sub_ab.src[1].src[0], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let b = AstNode::new(Op::Var("b".to_string()), vec![]);
        let mul_ab = a.clone() * b.clone();
        assert_eq!(mul_ab.op, Op::Mul);
        assert_eq!(mul_ab.src.len(), 2);
        assert_eq!(*mul_ab.src[0], a);
        assert_eq!(*mul_ab.src[1], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let b = AstNode::new(Op::Var("b".to_string()), vec![]);
        let div_ab = a.clone() / b.clone();
        assert_eq!(div_ab.op, Op::Mul); // div is implemented as a * (1/b)
        assert_eq!(div_ab.src.len(), 2);
        assert_eq!(*div_ab.src[0], a);
        assert_eq!(div_ab.src[1].op, Op::Recip);
        assert_eq!(*div_ab.src[1].src[0], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let b = AstNode::new(Op::Var("b".to_string()), vec![]);
        let rem_ab = a.clone() % b.clone();
        assert_eq!(rem_ab.op, Op::Rem);
        assert_eq!(rem_ab.src.len(), 2);
        assert_eq!(*rem_ab.src[0], a);
        assert_eq!(*rem_ab.src[1], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let b = AstNode::new(Op::Var("b".to_string()), vec![]);
        let max_ab = a.clone().max(b.clone());
        assert_eq!(max_ab.op, Op::Max);
        assert_eq!(max_ab.src.len(), 2);
        assert_eq!(*max_ab.src[0], a);
        assert_eq!(*max_ab.src[1], b);
    }

    #[test]
    fn test_complex_expression() {
        let a = AstNode::new(Op::Var("a".to_string()), vec![]);
        let b = AstNode::new(Op::Var("b".to_string()), vec![]);
        let c: AstNode = 2.0f64.into();

        // (a + b) * c
        let expr = (a.clone() + b.clone()) * c.clone();

        assert_eq!(expr.op, Op::Mul);
        assert_eq!(expr.src.len(), 2);
        assert_eq!(*expr.src[1], c);

        let add_expr = &expr.src[0];
        assert_eq!(add_expr.op, Op::Add);
        assert_eq!(add_expr.src.len(), 2);
        assert_eq!(*add_expr.src[0], a);
        assert_eq!(*add_expr.src[1], b);
    }

    #[test]
    fn test_partial_eq_ignores_id() {
        let node1 = AstNode::new(Op::Var("a".to_string()), vec![]);
        let node2 = AstNode::new(Op::Var("a".to_string()), vec![]);

        // IDs should be different
        assert_ne!(node1.id, node2.id);
        // But the nodes should be considered equal
        assert_eq!(node1, node2);

        let node3 = AstNode::new(Op::Var("b".to_string()), vec![]);
        assert_ne!(node1, node3);
    }
}
