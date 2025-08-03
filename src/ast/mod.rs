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

    // Tips: 除算 a/b は Mul(a, Recip(b)), 減算 a-b は Add(a, Neg(b)) のように表現する。
    // unary ops
    Neg,
    Recip,
    Sin,
    Sqrt,
    Log2,
    Exp2,
    Deref,
    Cast(DType), // convert dtype

    // binary ops
    Add,
    Mul,
    Max,
    Rem,
    LessThan,

    // Others
    Pack,         // Pack some values to tuple
    Index(usize), // Take n-th element from tuple
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

    pub fn cast(self, dtype: DType) -> Self {
        Self::new(Op::Cast(dtype.clone()), vec![Box::new(self)], dtype)
    }
}

macro_rules! impl_unary_op {
    ($op: ident, $fname: ident) => {
        impl AstNode {
            fn $fname(self: Self) -> Self {
                let dtype = &self.dtype;
                if !(dtype.is_real() || dtype.is_integer() || *dtype == DType::Any) {
                    panic!("Cannot apply {} to {:?}", stringify!($op), self.dtype)
                }
                AstNode::new(Op::$op, vec![Box::new(self.clone())], self.dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self) -> Self {
                let dtype = &self.dtype;
                if !(dtype.is_real() || *dtype == DType::Any) {
                    panic!("Cannot apply {} to {:?}", stringify!($op), self.dtype)
                }
                AstNode::new(Op::$op, vec![Box::new(self.clone())], self.dtype)
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
                let mut lhs = self;
                let mut rhs = other.into();

                if lhs.dtype != rhs.dtype {
                    // Attempt to promote types
                    let (l, r) = (&lhs.dtype, &rhs.dtype);
                    if l == &DType::Any {
                        lhs = lhs.cast(r.clone());
                    } else if r == &DType::Any {
                        rhs = rhs.cast(l.clone());
                    } else if l.is_real() && r.is_integer() {
                        rhs = rhs.cast(l.clone());
                    } else if l.is_integer() && r.is_real() {
                        lhs = lhs.cast(r.clone());
                    } else if l == &DType::F32 && r == &DType::F64 {
                        lhs = lhs.cast(DType::F64);
                    } else if l == &DType::F64 && r == &DType::F32 {
                        rhs = rhs.cast(DType::F64);
                    }
                }

                if lhs.dtype != rhs.dtype {
                    panic!(
                        "Cannot apply {} to {:?} and {:?}",
                        stringify!($op),
                        lhs.dtype,
                        rhs.dtype
                    );
                }

                let result_dtype = lhs.dtype.clone();
                AstNode::new(Op::$op, vec![Box::new(lhs), Box::new(rhs)], result_dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let mut lhs = self;
                let mut rhs = other.into();

                if lhs.dtype != rhs.dtype {
                    // Attempt to promote types
                    let (l, r) = (&lhs.dtype, &rhs.dtype);
                    if l == &DType::Any {
                        lhs = lhs.cast(r.clone());
                    } else if r == &DType::Any {
                        rhs = rhs.cast(l.clone());
                    } else if l.is_real() && r.is_integer() {
                        rhs = rhs.cast(l.clone());
                    } else if l.is_integer() && r.is_real() {
                        lhs = lhs.cast(r.clone());
                    } else if l == &DType::F32 && r == &DType::F64 {
                        lhs = lhs.cast(DType::F64);
                    } else if l == &DType::F64 && r == &DType::F32 {
                        rhs = rhs.cast(DType::F64);
                    }
                }

                if lhs.dtype != rhs.dtype {
                    panic!(
                        "Cannot apply {} to {:?} and {:?}",
                        stringify!($op),
                        lhs.dtype,
                        rhs.dtype
                    );
                }

                let result_dtype = lhs.dtype.clone();
                AstNode::new(Op::$op, vec![Box::new(lhs), Box::new(rhs)], result_dtype)
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
    USize,
    None,                  // void
    Ptr(Box<Self>),        // Pointer
    Vec(Box<Self>, usize), // Poiter of array
    Tuple(Vec<Self>),
    // for pattern matching
    Any,     // all types
    Natural, // natural number (includes 0)
    Integer, // integer
    Real,    // real number (actual implementation is float)
}

impl DType {
    pub fn is_real(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::Real)
    }

    pub fn is_natural(&self) -> bool {
        matches!(
            self,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::USize | DType::Natural
        )
    }

    pub fn is_integer(&self) -> bool {
        self.is_natural()
            || matches!(
                self,
                DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::Integer
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

macro_rules! impl_ast_assign_op {
    ($trait:ident, $fname:ident, $op_trait:ident, $op_fname:ident) => {
        impl<T> std::ops::$trait<T> for AstNode
        where
            T: Into<AstNode>,
        {
            fn $fname(&mut self, rhs: T) {
                *self = std::ops::$op_trait::$op_fname(self.clone(), rhs.into());
            }
        }
    };
}

impl_ast_assign_op!(AddAssign, add_assign, Add, add);
impl_ast_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_ast_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_ast_assign_op!(DivAssign, div_assign, Div, div);
impl_ast_assign_op!(RemAssign, rem_assign, Rem, rem);

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
        let a = AstNode::var("a").with_type(DType::F64);
        let b = AstNode::var("b").with_type(DType::F64);
        let c: AstNode = 2.0f64.into();

        // (a + b) * c
        let expr = (a.clone() + b.clone()) * c.clone();

        assert_eq!(expr.op, Op::Mul);
        assert_eq!(expr.src.len(), 2);
        assert_eq!(*expr.src[1], c);

        let add_expr = &*expr.src[0];
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

        // USize
        assert!(DType::Natural.matches(&DType::USize));
        assert!(DType::Integer.matches(&DType::USize));
        assert!(!DType::Real.matches(&DType::USize));
    }

    #[test]
    fn test_op_type_check_ok() {
        let f1 = AstNode::var("f1").with_type(DType::F32);
        let f2 = AstNode::var("f2").with_type(DType::F64);
        let i1 = AstNode::var("i1").with_type(DType::I32);

        // Real + Real -> Real
        let add_ff = f1.clone() + f2.clone();
        assert!(add_ff.dtype.is_real());

        // Real + Integer -> Real
        let add_fi = f1.clone() + i1.clone();
        assert!(add_fi.dtype.is_real());

        // Neg on Real
        let neg_f = -f1;
        assert!(neg_f.dtype.is_real());
    }

    #[test]
    #[should_panic]
    fn test_op_type_check_panic_add() {
        let p1 = AstNode::var("p1").with_type(DType::Ptr(Box::new(DType::F32)));
        let i1 = AstNode::var("i1").with_type(DType::I32);
        // Ptr + Integer should panic
        let _ = p1 + i1;
    }

    #[test]
    #[should_panic]
    fn test_op_type_check_panic_neg() {
        let p1 = AstNode::var("p1").with_type(DType::Ptr(Box::new(DType::F32)));
        // Neg on Ptr should panic
        let _ = -p1;
    }

    #[test]
    fn test_implicit_cast() {
        let i1 = AstNode::var("i1").with_type(DType::I32);
        let f1 = AstNode::var("f1").with_type(DType::F32);

        // i1 + f1 should result in Cast(I32 as F32) + F32
        let result = i1.clone() + f1.clone();

        assert_eq!(result.op, Op::Add);
        assert_eq!(result.dtype, DType::F32);

        let lhs = &*result.src[0];
        let rhs = &*result.src[1];

        // Check that the integer was cast to float
        assert_eq!(lhs.op, Op::Cast(DType::F32));
        assert_eq!(lhs.dtype, DType::F32);
        assert_eq!(*lhs.src[0], i1); // Original i1 inside the cast

        // Check that the float remains unchanged
        assert_eq!(*rhs, f1);
    }

    #[test]
    fn test_ast_add_assign() {
        let mut a = AstNode::var("a").with_type(DType::F32);
        let b = AstNode::var("b").with_type(DType::F32);
        let c = a.clone() + b.clone();
        a += b;
        assert_eq!(a, c);
    }

    #[test]
    fn test_ast_sub_assign() {
        let mut a = AstNode::var("a").with_type(DType::F32);
        let b = AstNode::var("b").with_type(DType::F32);
        let c = a.clone() - b.clone();
        a -= b;
        assert_eq!(a, c);
    }
}
