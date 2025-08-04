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

macro_rules! impl_from_numeric_for_expr {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Expr {
                fn from(val: $t) -> Self {
                    Expr::Const(val as i64)
                }
            }
        )*
    };
}

impl_from_numeric_for_expr!(i8, i16, i32, isize, u8, u16, u32, u64, usize, i128, u128);

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

macro_rules! impl_i64_binary_op {
    ($trait:ident, $fname:ident, $variant:expr) => {
        impl $trait<Expr> for i64 {
            type Output = Expr;
            fn $fname(self, rhs: Expr) -> Self::Output {
                $variant(Box::new(Expr::from(self)), Box::new(rhs))
            }
        }
    };
}

impl_i64_binary_op!(Add, add, Expr::Add);
impl_i64_binary_op!(Sub, sub, Expr::Sub);
impl_i64_binary_op!(Mul, mul, Expr::Mul);
impl_i64_binary_op!(Div, div, Expr::Div);
impl_i64_binary_op!(Rem, rem, Expr::Rem);

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

impl From<AstNode> for Expr {
    fn from(node: AstNode) -> Self {
        match node.op {
            crate::ast::Op::Var(s) => Expr::Var(s),
            crate::ast::Op::Const(c) => match c {
                crate::ast::Const::I64(v) => Expr::Const(v),
                _ => panic!("Cannot convert this const type to Expr"),
            },
            // This is a simplified conversion. More complex AST nodes might not have a direct Expr equivalent.
            _ => panic!("Cannot convert this AstNode to Expr"),
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
            Expr::Mul(
                Box::new(Expr::Add(Box::new(n), Box::new(Expr::Const(1)))),
                Box::new(Expr::Const(2))
            )
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
        assert_eq!((0 + n.clone()).simplify(), n.clone());
        assert_eq!((Expr::from(2) + 3).simplify(), Expr::from(5));

        // Sub
        assert_eq!((n.clone() - 0).simplify(), n.clone());
        assert_eq!((n.clone() - n.clone()).simplify(), Expr::Const(0));
        assert_eq!((Expr::from(5) - 3).simplify(), Expr::from(2));

        // Mul
        assert_eq!((n.clone() * 1).simplify(), n.clone());
        assert_eq!((1 * n.clone()).simplify(), n.clone());
        assert_eq!((n.clone() * 0).simplify(), Expr::Const(0));
        assert_eq!((0 * n.clone()).simplify(), Expr::Const(0));
        assert_eq!((Expr::from(2) * 3).simplify(), Expr::from(6));

        // Div
        assert_eq!((n.clone() / 1).simplify(), n.clone());
        assert_eq!((n.clone() / n.clone()).simplify(), Expr::Const(1));
        assert_eq!((0 / n.clone()).simplify(), Expr::Const(0));
        assert_eq!((Expr::from(6) / 3).simplify(), Expr::from(2));

        // Rem
        assert_eq!((n.clone() % 1).simplify(), Expr::Const(0));
        assert_eq!((n.clone() % n.clone()).simplify(), Expr::Const(0));
        assert_eq!((0 % n.clone()).simplify(), Expr::Const(0));
        assert_eq!((Expr::from(7) % 3).simplify(), Expr::from(1));

        // Recursive
        assert_eq!(((n.clone() + 0) * 1).simplify(), n.clone());
        assert_eq!(((n.clone() * 1) + (m.clone() * 0)).simplify(), n.clone());
        assert_eq!(((n.clone() + 1) - 1).simplify(), n.clone());
    }

    #[test]
    fn test_from_numeric_for_expr() {
        assert_eq!(Expr::from(10i8), Expr::Const(10));
        assert_eq!(Expr::from(10u8), Expr::Const(10));
        assert_eq!(Expr::from(10i16), Expr::Const(10));
        assert_eq!(Expr::from(10u16), Expr::Const(10));
        assert_eq!(Expr::from(10i32), Expr::Const(10));
        assert_eq!(Expr::from(10u32), Expr::Const(10));
        assert_eq!(Expr::from(10i64), Expr::Const(10));
        assert_eq!(Expr::from(10u64), Expr::Const(10));
        assert_eq!(Expr::from(10isize), Expr::Const(10));
        assert_eq!(Expr::from(10usize), Expr::Const(10));
        assert_eq!(Expr::from(10i128), Expr::Const(10));
        assert_eq!(Expr::from(10u128), Expr::Const(10));
    }
}
