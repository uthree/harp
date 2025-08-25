use crate::graph::shape::Expr as ShapeExpr;
pub mod pattern;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,   // float
    Usize, // size_t
    Isize, // ssize_t
    Void,

    Ptr(Box<Self>, ShapeExpr), // pointer
    Vec(Box<Self>, usize),     // fixed-size array (for SIMD vectorization)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstLiteral {
    F32(f32),
    Usize(usize),
    Isize(isize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    Const(ConstLiteral), // constant value
    Var(String),         // get value from variable
    Cast(DType),         // convert another type

    // numeric ops
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    Neg(Box<Self>),
    Recip(Box<Self>),
    Sin(Box<Self>),
    Sqrt(Box<Self>),
    Log2(Box<Self>),
    Exp2(Box<Self>),
    CallFunction(Vec<Self>),

    // statements
    Range {
        // Forループ
        counter_name: String, // ループカウンタの変数名
        max: Box<Self>,       // ループ回数
        body: Vec<Self>,
    },

    Declare {
        name: String,
        dtype: DType,
        constant: bool,
    }, // declare new (local) variable

    Drop(String), // drop (local) variable explicitly

    Barrier,

    // for pattern matching
    Capture(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    name: String,
    body: Vec<AstNode>,
    // TODO: arguments, return values
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    functions: Vec<Function>,
    entry_point: String,
}

macro_rules! impl_from_num_for_astnode {
    ($(($t:ty, $v: ident)),*) => {
        $(
            impl From<$t> for AstNode {
                fn from(n: $t) -> Self {
                    AstNode::Const(ConstLiteral::$v(n))
                }
            }
        )*
    };
}
impl_from_num_for_astnode!((usize, Usize), (isize, Isize), (f32, F32));

impl From<ConstLiteral> for AstNode {
    fn from(c: ConstLiteral) -> Self {
        AstNode::Const(c)
    }
}

macro_rules! impl_astnode_binary_op {
    ($trait:ident, $fname:ident, $variant:ident) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: T) -> Self::Output {
                AstNode::$variant(Box::new(self), Box::new(rhs.into()))
            }
        }
    };
}

impl_astnode_binary_op!(Add, add, Add);
impl_astnode_binary_op!(Mul, mul, Mul);
impl_astnode_binary_op!(Rem, rem, Rem);

impl<T: Into<AstNode>> Sub<T> for AstNode {
    type Output = AstNode;
    fn sub(self, rhs: T) -> Self::Output {
        self + AstNode::Neg(Box::new(rhs.into()))
    }
}

impl<T: Into<AstNode>> Div<T> for AstNode {
    type Output = AstNode;
    fn div(self, rhs: T) -> Self::Output {
        self + AstNode::Recip(Box::new(rhs.into()))
    }
}

macro_rules! impl_expr_assign_op {
    ($trait:ident, $fname:ident, $op:tt) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
            fn $fname(&mut self, rhs: T) {
                *self = self.clone() $op rhs.into();
            }
        }
    };
}

impl_expr_assign_op!(AddAssign, add_assign, +);
impl_expr_assign_op!(SubAssign, sub_assign, -);
impl_expr_assign_op!(MulAssign, mul_assign, *);
impl_expr_assign_op!(DivAssign, div_assign, /);
impl_expr_assign_op!(RemAssign, rem_assign, %);

impl Neg for AstNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        AstNode::Neg(Box::new(self))
    }
}

macro_rules! impl_astnode_unary_op {
    ($fname:ident, $variant:ident) => {
        impl AstNode {
            fn $fname(self) -> Self {
                AstNode::$variant(Box::new(self))
            }
        }
    };
}

impl_astnode_unary_op!(recip, Recip);
impl_astnode_unary_op!(sin, Sin);
impl_astnode_unary_op!(sqrt, Sqrt);
impl_astnode_unary_op!(exp2, Exp2);
impl_astnode_unary_op!(log2, Log2);
