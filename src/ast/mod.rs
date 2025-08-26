use crate::graph::shape::Expr as ShapeExpr;
pub mod pattern;
use std::ops::{
    Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
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

        impl $trait<&AstNode> for &AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: &AstNode) -> Self::Output {
                AstNode::$variant(Box::new(self.clone()), Box::new(rhs.clone()))
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
            pub fn $fname(self) -> Self {
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

impl AstNode {
    pub fn capture(n: usize) -> AstNode {
        AstNode::Capture(n)
    }

    pub fn children(&self) -> Vec<&AstNode> {
        match self {
            AstNode::Const(_) => vec![],
            AstNode::Var(_) => vec![],
            AstNode::Cast(_) => vec![],
            AstNode::Add(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Mul(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Max(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Rem(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Neg(n) => vec![n.as_ref()],
            AstNode::Recip(n) => vec![n.as_ref()],
            AstNode::Sin(n) => vec![n.as_ref()],
            AstNode::Sqrt(n) => vec![n.as_ref()],
            AstNode::Log2(n) => vec![n.as_ref()],
            AstNode::Exp2(n) => vec![n.as_ref()],
            AstNode::CallFunction(nodes) => nodes.iter().collect(),
            AstNode::Range { max, body, .. } => {
                let mut children = vec![max.as_ref()];
                children.extend(body.iter());
                children
            }
            AstNode::Declare { .. } => vec![],
            AstNode::Drop(_) => vec![],
            AstNode::Barrier => vec![],
            AstNode::Capture(_) => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(1.0f32, AstNode::Const(ConstLiteral::F32(1.0)))]
    #[case(42usize, AstNode::Const(ConstLiteral::Usize(42)))]
    #[case(-10isize, AstNode::Const(ConstLiteral::Isize(-10)))]
    fn test_from_numeric_literals(#[case] input: impl Into<AstNode>, #[case] expected: AstNode) {
        assert_eq!(input.into(), expected);
    }

    #[test]
    fn test_addition() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a + b;
        assert_eq!(
            expr,
            AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_subtraction() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a - b;
        assert_eq!(
            expr,
            AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Neg(Box::new(AstNode::Const(ConstLiteral::F32(
                    2.0
                )))))
            )
        );
    }

    #[test]
    fn test_multiplication() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a * b;
        assert_eq!(
            expr,
            AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_division() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a / b;
        assert_eq!(
            expr,
            AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Recip(Box::new(AstNode::Const(ConstLiteral::F32(
                    2.0
                )))))
            )
        );
    }

    #[test]
    fn test_remainder() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a % b;
        assert_eq!(
            expr,
            AstNode::Rem(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_negation() {
        let a = AstNode::Var("a".to_string());
        let expr = -a;
        assert_eq!(expr, AstNode::Neg(Box::new(AstNode::Var("a".to_string()))));
    }

    #[test]
    fn test_unary_ops() {
        let a = AstNode::Var("a".to_string());
        assert_eq!(
            a.clone().recip(),
            AstNode::Recip(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().sin(),
            AstNode::Sin(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().sqrt(),
            AstNode::Sqrt(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().exp2(),
            AstNode::Exp2(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.log2(),
            AstNode::Log2(Box::new(AstNode::Var("a".to_string())))
        );
    }

    #[test]
    fn test_complex_expression() {
        let a = AstNode::Var("a".to_string());
        let b = AstNode::Var("b".to_string());
        let c = 3.0f32;
        // -(a + b) * c
        let expr = -(a.clone() + b.clone()) * c;
        assert_eq!(
            expr,
            AstNode::Mul(
                Box::new(AstNode::Neg(Box::new(AstNode::Add(
                    Box::new(AstNode::Var("a".to_string())),
                    Box::new(AstNode::Var("b".to_string()))
                )))),
                Box::new(AstNode::Const(ConstLiteral::F32(3.0)))
            )
        );
    }
}
