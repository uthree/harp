pub mod pattern;

use std::{boxed::Box, cell::Cell};

thread_local! {
    static NEXT_ID: Cell<usize> = const { Cell::new(0) };
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
    Capture(usize, DType),

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

    pub fn capture(id: usize, dtype: DType) -> Self {
        Self::new(Op::Capture(id, dtype.clone()), vec![], dtype)
    }

    pub fn var(name: &str) -> Self {
        Self::new(Op::Var(name.to_string()), vec![], DType::Any)
    }

    pub fn with_type(self, dtype: DType) -> Self {
        Self::new(self.op, self.src, dtype)
    }
}

macro_rules! impl_unary_op {
    ($op: ident, $fname: ident, $dtype: expr) => {
        impl AstNode {
            fn $fname(self: Self) -> Self {
                AstNode::new(Op::$op, vec![Box::new(self)], $dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident, $dtype: expr) => {
        impl AstNode {
            pub fn $fname(self: Self) -> Self {
                AstNode::new(Op::$op, vec![Box::new(self)], $dtype)
            }
        }
    };
}

impl_unary_op!(Neg, neg_, DType::Any);
impl_unary_op!(Recip, recip, DType::Any);
impl_unary_op!(pub, Sqrt, sqrt, DType::Any);
impl_unary_op!(pub, Sin, sin, DType::Any);
impl_unary_op!(pub, Log2, log2, DType::Any);
impl_unary_op!(pub, Exp2, exp2, DType::Any);

macro_rules! impl_binary_op {
    ($op: ident, $fname: ident, $dtype: expr) => {
        impl AstNode {
            fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let other = other.into();
                AstNode::new(Op::$op, vec![Box::new(self), Box::new(other)], $dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident, $dtype: expr) => {
        impl AstNode {
            pub fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let other = other.into();
                AstNode::new(Op::$op, vec![Box::new(self), Box::new(other)], $dtype)
            }
        }
    };
}

impl_binary_op!(Add, add_, DType::Any);
impl_binary_op!(Mul, mul_, DType::Any);
impl_binary_op!(pub, Max, max, DType::Any);
impl_binary_op!(Rem, rem_, DType::Any);

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
    None,                  // void
    Ptr(Box<Self>),        // Pointer
    Vec(Box<Self>, usize), // Poiter of array
    // for pattern matching
    Any,     // all types
    Natural, // natural number (includes 0)
    Integer, // integer
    Real,    // real number (actual implementation is float)
}

impl DType {
    pub fn is_real(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    pub fn is_natural(&self) -> bool {
        matches!(self, DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
        )
    }

    pub fn matches(&self, other: &DType) -> bool {
        if self == other {
            return true;
        }
        match self {
            DType::Any => true,
            DType::Real => other.is_real(),
            DType::Natural => other.is_natural(),
            DType::Integer => other.is_integer(),
            DType::Ptr(a) => {
                if let DType::Ptr(b) = other {
                    a.matches(b)
                } else {
                    false
                }
            }
            DType::Vec(a, ..) => {
                if let DType::Vec(b, ..) = other {
                    a.matches(b)
                } else {
                    false
                }
            }
            _ => false,
        }
    }
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
                let c = Const::$variant(v);
                AstNode::new(Op::Const(c), vec![], c.dtype())
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

    use crate::ast::{AstNode, DType, Op};
    #[test]
    fn test_unary_ops() {
        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);

        let neg_a = -a.clone();
        assert_eq!(neg_a.op, Op::Neg);
        assert_eq!(neg_a.src.len(), 1);
        assert_eq!(*neg_a.src[0], a);

        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let sqrt_a = a.clone().sqrt();
        assert_eq!(sqrt_a.op, Op::Sqrt);
        assert_eq!(sqrt_a.src.len(), 1);
        assert_eq!(*sqrt_a.src[0], a);

        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let sin_a = a.clone().sin();
        assert_eq!(sin_a.op, Op::Sin);
        assert_eq!(sin_a.src.len(), 1);
        assert_eq!(*sin_a.src[0], a);
    }

    #[test]
    fn test_binary_ops() {
        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);

        let add_ab = a.clone() + b.clone();
        assert_eq!(add_ab.op, Op::Add);
        assert_eq!(add_ab.src.len(), 2);
        assert_eq!(*add_ab.src[0], a);
        assert_eq!(*add_ab.src[1], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);
        let sub_ab = a.clone() - b.clone();
        assert_eq!(sub_ab.op, Op::Add); // sub is implemented as a + (-b)
        assert_eq!(sub_ab.src.len(), 2);
        assert_eq!(*sub_ab.src[0], a);
        assert_eq!(sub_ab.src[1].op, Op::Neg);
        assert_eq!(*sub_ab.src[1].src[0], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);
        let mul_ab = a.clone() * b.clone();
        assert_eq!(mul_ab.op, Op::Mul);
        assert_eq!(mul_ab.src.len(), 2);
        assert_eq!(*mul_ab.src[0], a);
        assert_eq!(*mul_ab.src[1], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);
        let div_ab = a.clone() / b.clone();
        assert_eq!(div_ab.op, Op::Mul); // div is implemented as a * (1/b)
        assert_eq!(div_ab.src.len(), 2);
        assert_eq!(*div_ab.src[0], a);
        assert_eq!(div_ab.src[1].op, Op::Recip);
        assert_eq!(*div_ab.src[1].src[0], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);
        let rem_ab = a.clone() % b.clone();
        assert_eq!(rem_ab.op, Op::Rem);
        assert_eq!(rem_ab.src.len(), 2);
        assert_eq!(*rem_ab.src[0], a);
        assert_eq!(*rem_ab.src[1], b);

        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);
        let max_ab = a.clone().max(b.clone());
        assert_eq!(max_ab.op, Op::Max);
        assert_eq!(max_ab.src.len(), 2);
        assert_eq!(*max_ab.src[0], a);
        assert_eq!(*max_ab.src[1], b);
    }

    #[test]
    fn test_complex_expression() {
        let a = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);
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
        let node1 = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);
        let node2 = AstNode::new(Op::Var("a".to_string()), vec![], DType::Any);

        // IDs should be different
        assert_ne!(node1.id, node2.id);
        // But the nodes should be considered equal
        assert_eq!(node1, node2);

        let node3 = AstNode::new(Op::Var("b".to_string()), vec![], DType::Any);
        assert_ne!(node1, node3);
    }

    #[test]
    fn test_dtype_matches() {
        use super::DType;

        // Exact matches
        assert!(DType::F32.matches(&DType::F32));
        assert!(!DType::F32.matches(&DType::F64));

        // Any
        assert!(DType::Any.matches(&DType::F32));
        assert!(DType::Any.matches(&DType::I64));
        assert!(DType::Any.matches(&DType::U8));
        assert!(DType::Any.matches(&DType::Ptr(Box::new(DType::F32))));

        // Real
        assert!(DType::Real.matches(&DType::F32));
        assert!(DType::Real.matches(&DType::F64));
        assert!(!DType::Real.matches(&DType::I32));

        // Natural
        assert!(DType::Natural.matches(&DType::U8));
        assert!(DType::Natural.matches(&DType::U64));
        assert!(!DType::Natural.matches(&DType::I8));
        assert!(!DType::Natural.matches(&DType::F32));

        // Integer
        assert!(DType::Integer.matches(&DType::I32));
        assert!(DType::Integer.matches(&DType::U16));
        assert!(!DType::Integer.matches(&DType::F64));

        // Pointer
        let p_f32 = DType::Ptr(Box::new(DType::F32));
        let p_f64 = DType::Ptr(Box::new(DType::F64));
        let p_any = DType::Ptr(Box::new(DType::Any));
        assert!(p_f32.matches(&p_f32));
        assert!(!p_f32.matches(&p_f64));
        assert!(p_any.matches(&p_f32));
        assert!(!p_f32.matches(&p_any)); // A specific type does not match a general one
    }
}
