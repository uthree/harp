use crate::ast::{AstNode, DType};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Const(i64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Rem(Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_string())
    }

    pub fn simplify(self) -> Self {
        match self {
            Expr::Add(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(0), e) | (e, Expr::Const(0)) => e,
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l + r),
                    (l, r) => l + r,
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (e, Expr::Const(0)) => e,
                    (l, r) if l == r => Expr::Const(0),
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l - r),
                    (Expr::Add(a, b), r) if *b == r => *a,
                    (Expr::Add(a, b), r) if *a == r => *b,
                    (l, r) => l - r,
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(0), _) | (_, Expr::Const(0)) => Expr::Const(0),
                    (Expr::Const(1), e) | (e, Expr::Const(1)) => e,
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l * r),
                    (l, r) => l * r,
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (_, Expr::Const(0)) => panic!("division by zero"),
                    (e, Expr::Const(1)) => e,
                    (l, r) if l == r => Expr::Const(1),
                    (Expr::Const(0), _) => Expr::Const(0),
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l / r),
                    (l, r) => l / r,
                }
            }
            Expr::Rem(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (_, Expr::Const(0)) => panic!("division by zero"),
                    (_, Expr::Const(1)) => Expr::Const(0),
                    (l, r) if l == r => Expr::Const(0),
                    (Expr::Const(0), _) => Expr::Const(0),
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l % r),
                    (l, r) => l % r,
                }
            }
            _ => self,
        }
    }
}

impl From<i64> for Expr {
    fn from(val: i64) -> Self {
        Expr::Const(val)
    }
}

macro_rules! impl_expr_binary_op {
    ($trait:ident, $fname:ident, $variant:expr) => {
        impl<T: Into<Expr>> $trait<T> for Expr {
            type Output = Expr;
            fn $fname(self, rhs: T) -> Self::Output {
                $variant(Box::new(self), Box::new(rhs.into()))
            }
        }
    };
}

impl_expr_binary_op!(Add, add, Expr::Add);
impl_expr_binary_op!(Sub, sub, Expr::Sub);
impl_expr_binary_op!(Mul, mul, Expr::Mul);
impl_expr_binary_op!(Div, div, Expr::Div);
impl_expr_binary_op!(Rem, rem, Expr::Rem);

macro_rules! impl_expr_assign_op {
    ($trait:ident, $fname:ident, $op_trait:ident, $op_fname:ident) => {
        impl<T> $trait<T> for Expr
        where
            T: Into<Expr>,
        {
            fn $fname(&mut self, rhs: T) {
                *self = $op_trait::$op_fname(self.clone(), rhs.into());
            }
        }
    };
}

impl_expr_assign_op!(AddAssign, add_assign, Add, add);
impl_expr_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_expr_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_expr_assign_op!(DivAssign, div_assign, Div, div);
impl_expr_assign_op!(RemAssign, rem_assign, Rem, rem);

impl From<Expr> for AstNode {
    fn from(expr: Expr) -> Self {
        match expr {
            Expr::Const(c) => (c).into(),
            Expr::Var(s) => AstNode::var(&s).with_type(DType::I64),
            Expr::Add(l, r) => AstNode::from(*l) + AstNode::from(*r),
            Expr::Sub(l, r) => AstNode::from(*l) - AstNode::from(*r),
            Expr::Mul(l, r) => AstNode::from(*l) * AstNode::from(*r),
            Expr::Div(l, r) => AstNode::from(*l) / AstNode::from(*r),
            Expr::Rem(l, r) => AstNode::from(*l) % AstNode::from(*r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Op;

    #[test]
    fn test_expr_ops() {
        let n = Expr::var("N");
        let expr = (n.clone() + 1) * 2;
        assert_eq!(
            expr,
            Expr::Mul(Box::new(Expr::Add(Box::new(n), Box::new(Expr::Const(1)))), Box::new(Expr::Const(2)))
        );
    }

    #[test]
    fn test_expr_assign_ops() {
        let mut n = Expr::var("N");
        let original_n = n.clone();
        n += 1;
        assert_eq!(n, Expr::Add(Box::new(original_n), Box::new(Expr::Const(1))));
    }

    #[test]
    fn test_expr_to_ast() {
        let n = Expr::var("N");
        let expr = (n + 1) * 2;
        let ast: AstNode = expr.into();

        assert_eq!(ast.op, Op::Mul);
        assert_eq!(ast.dtype, DType::I64);

        let lhs = &*ast.src[0];
        let rhs = &*ast.src[1];

        assert_eq!(lhs.op, Op::Add);
        assert_eq!(lhs.dtype, DType::I64);
        assert_eq!(rhs.op, Op::Const(crate::ast::Const::I64(2)));
    }

    #[test]
    fn test_expr_simplify() {
        let n = Expr::var("N");
        let m = Expr::var("M");

        // Add
        assert_eq!((n.clone() + 0).simplify(), n.clone());
        assert_eq!((0.into_expr() + n.clone()).simplify(), n.clone());
        assert_eq!((2 + 3).into_expr().simplify(), 5.into_expr());

        // Sub
        assert_eq!((n.clone() - 0).simplify(), n.clone());
        assert_eq!((n.clone() - n.clone()).simplify(), Expr::Const(0));
        assert_eq!((5 - 3).into_expr().simplify(), 2.into_expr());

        // Mul
        assert_eq!((n.clone() * 1).simplify(), n.clone());
        assert_eq!((1.into_expr() * n.clone()).simplify(), n.clone());
        assert_eq!((n.clone() * 0).simplify(), Expr::Const(0));
        assert_eq!((0.into_expr() * n.clone()).simplify(), Expr::Const(0));
        assert_eq!((2 * 3).into_expr().simplify(), 6.into_expr());

        // Div
        assert_eq!((n.clone() / 1).simplify(), n.clone());
        assert_eq!((n.clone() / n.clone()).simplify(), Expr::Const(1));
        assert_eq!((0.into_expr() / n.clone()).simplify(), Expr::Const(0));
        assert_eq!((6 / 3).into_expr().simplify(), 2.into_expr());

        // Rem
        assert_eq!((n.clone() % 1).simplify(), Expr::Const(0));
        assert_eq!((n.clone() % n.clone()).simplify(), Expr::Const(0));
        assert_eq!((0.into_expr() % n.clone()).simplify(), Expr::Const(0));
        assert_eq!((7 % 3).into_expr().simplify(), 1.into_expr());

        // Recursive
        assert_eq!(((n.clone() + 0) * 1).simplify(), n.clone());
        assert_eq!(((n.clone() * 1) + (m.clone() * 0)).simplify(), n.clone());
        assert_eq!(((n.clone() + 1) - 1).simplify(), n.clone());
    }

    trait IntoExpr {
        fn into_expr(self) -> Expr;
    }

    impl IntoExpr for i64 {
        fn into_expr(self) -> Expr {
            self.into()
        }
    }
}