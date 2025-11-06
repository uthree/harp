use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    // 定数と変数
    Const(isize),
    Var(String),

    // 算術演算
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
}

impl From<Expr> for crate::ast::AstNode {
    fn from(expr: Expr) -> Self {
        use crate::ast::{AstNode, Literal};

        // 変換前にsimplifyして可読性を向上
        let expr = expr.simplify();
        match expr {
            Expr::Const(c) => AstNode::Const(Literal::Isize(c)),
            Expr::Var(s) => {
                // 変数を直接ASTのVarに変換
                AstNode::Var(s)
            }
            Expr::Add(l, r) => AstNode::Add(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Sub(l, r) => {
                // a - b = a + (-b)
                let left: AstNode = (*l).into();
                let right: AstNode = (*r).into();
                left + (-right)
            }
            Expr::Mul(l, r) => AstNode::Mul(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Div(l, r) => {
                // a / b = a * recip(b)
                let left: AstNode = (*l).into();
                let right: AstNode = (*r).into();
                left * crate::ast::helper::recip(right)
            }
            Expr::Rem(l, r) => AstNode::Rem(Box::new((*l).into()), Box::new((*r).into())),
        }
    }
}

impl Expr {
    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Const(0))
    }

    pub fn is_one(&self) -> bool {
        matches!(self, Expr::Const(1))
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
                    // Simplify double negation: 0 - (a - b) -> b - a
                    (Expr::Const(0), Expr::Sub(a, b)) => (*b - *a).simplify(),
                    (l, r) => l - r,
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(0), _) | (_, Expr::Const(0)) => Expr::Const(0),
                    (Expr::Const(1), e) | (e, Expr::Const(1)) => e,
                    (Expr::Const(-1), e) => (-e).simplify(),
                    (e, Expr::Const(-1)) => (-e).simplify(),
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

macro_rules! impl_from_integer_for_expr {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Expr {
                fn from(n: $t) -> Self {
                    Expr::Const(n as isize)
                }
            }
        )*
    };
}

impl_from_integer_for_expr!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        Expr::Var(s.to_string())
    }
}

impl From<String> for Expr {
    fn from(s: String) -> Self {
        Expr::Var(s)
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
    ($trait:ident, $fname:ident, $op:tt) => {
        impl<T: Into<Expr>> $trait<T> for Expr {
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

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Expr::from(0isize) - self
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(n) => write!(f, "{}", n),
            Expr::Var(s) => write!(f, "{}", s),
            Expr::Add(lhs, rhs) => {
                // Add parentheses only when necessary
                let needs_parens_lhs = matches!(**lhs, Expr::Sub(_, _));
                let needs_parens_rhs = matches!(**rhs, Expr::Sub(_, _));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " + ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Sub(lhs, rhs) => {
                let needs_parens_rhs = !matches!(**rhs, Expr::Const(_) | Expr::Var(_));

                write!(f, "{}", lhs)?;
                write!(f, " - ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Mul(lhs, rhs) => {
                let needs_parens_lhs = matches!(**lhs, Expr::Add(_, _) | Expr::Sub(_, _));
                let needs_parens_rhs = matches!(**rhs, Expr::Add(_, _) | Expr::Sub(_, _));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " * ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Div(lhs, rhs) => {
                let needs_parens_lhs = matches!(**lhs, Expr::Add(_, _) | Expr::Sub(_, _));
                let needs_parens_rhs = !matches!(**rhs, Expr::Const(_) | Expr::Var(_));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " / ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Rem(lhs, rhs) => {
                let needs_parens_lhs = matches!(**lhs, Expr::Add(_, _) | Expr::Sub(_, _));
                let needs_parens_rhs = !matches!(**rhs, Expr::Const(_) | Expr::Var(_));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " % ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_creation() {
        let expr = Expr::Const(42);
        assert_eq!(expr, Expr::Const(42));
    }

    #[test]
    fn test_var_creation() {
        let expr = Expr::Var("x".to_string());
        assert_eq!(expr, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_from_integer() {
        let expr = Expr::from(10isize);
        assert_eq!(expr, Expr::Const(10));

        let expr = Expr::from(5usize);
        assert_eq!(expr, Expr::Const(5));
    }

    #[test]
    fn test_from_str() {
        let expr = Expr::from("x");
        assert_eq!(expr, Expr::Var("x".to_string()));

        let expr = Expr::from("batch_size".to_string());
        assert_eq!(expr, Expr::Var("batch_size".to_string()));
    }

    #[test]
    fn test_is_zero() {
        assert!(Expr::Const(0).is_zero());
        assert!(!Expr::Const(1).is_zero());
        assert!(!Expr::Var("x".to_string()).is_zero());
    }

    #[test]
    fn test_is_one() {
        assert!(Expr::Const(1).is_one());
        assert!(!Expr::Const(0).is_one());
        assert!(!Expr::Var("x".to_string()).is_one());
    }

    #[test]
    fn test_add_operator() {
        let a = Expr::Const(2);
        let b = Expr::Const(3);
        let sum = a + b;
        assert_eq!(
            sum,
            Expr::Add(Box::new(Expr::Const(2)), Box::new(Expr::Const(3)))
        );
    }

    #[test]
    fn test_sub_operator() {
        let a = Expr::Const(5);
        let b = Expr::Const(3);
        let diff = a - b;
        assert_eq!(
            diff,
            Expr::Sub(Box::new(Expr::Const(5)), Box::new(Expr::Const(3)))
        );
    }

    #[test]
    fn test_mul_operator() {
        let a = Expr::Const(4);
        let b = Expr::Const(5);
        let prod = a * b;
        assert_eq!(
            prod,
            Expr::Mul(Box::new(Expr::Const(4)), Box::new(Expr::Const(5)))
        );
    }

    #[test]
    fn test_div_operator() {
        let a = Expr::Const(10);
        let b = Expr::Const(2);
        let quot = a / b;
        assert_eq!(
            quot,
            Expr::Div(Box::new(Expr::Const(10)), Box::new(Expr::Const(2)))
        );
    }

    #[test]
    fn test_rem_operator() {
        let a = Expr::Const(10);
        let b = Expr::Const(3);
        let rem = a % b;
        assert_eq!(
            rem,
            Expr::Rem(Box::new(Expr::Const(10)), Box::new(Expr::Const(3)))
        );
    }

    #[test]
    fn test_neg_operator() {
        let a = Expr::Const(5);
        let neg_a = -a;
        assert_eq!(
            neg_a,
            Expr::Sub(Box::new(Expr::Const(0)), Box::new(Expr::Const(5)))
        );
    }

    #[test]
    fn test_simplify_add_zero() {
        // 0 + x = x
        let expr = Expr::Const(0) + Expr::Var("x".to_string());
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Var("x".to_string()));

        // x + 0 = x
        let expr = Expr::Var("x".to_string()) + Expr::Const(0);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_simplify_add_const() {
        // 2 + 3 = 5
        let expr = Expr::Const(2) + Expr::Const(3);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(5));
    }

    #[test]
    fn test_simplify_sub_zero() {
        // x - 0 = x
        let expr = Expr::Var("x".to_string()) - Expr::Const(0);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_simplify_sub_self() {
        // x - x = 0
        let x = Expr::Var("x".to_string());
        let expr = x.clone() - x;
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(0));
    }

    #[test]
    fn test_simplify_sub_const() {
        // 5 - 3 = 2
        let expr = Expr::Const(5) - Expr::Const(3);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(2));
    }

    #[test]
    fn test_simplify_mul_zero() {
        // 0 * x = 0
        let expr = Expr::Const(0) * Expr::Var("x".to_string());
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(0));

        // x * 0 = 0
        let expr = Expr::Var("x".to_string()) * Expr::Const(0);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(0));
    }

    #[test]
    fn test_simplify_mul_one() {
        // 1 * x = x
        let expr = Expr::Const(1) * Expr::Var("x".to_string());
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Var("x".to_string()));

        // x * 1 = x
        let expr = Expr::Var("x".to_string()) * Expr::Const(1);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_simplify_mul_const() {
        // 3 * 4 = 12
        let expr = Expr::Const(3) * Expr::Const(4);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(12));
    }

    #[test]
    fn test_simplify_div_one() {
        // x / 1 = x
        let expr = Expr::Var("x".to_string()) / Expr::Const(1);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_simplify_div_self() {
        // x / x = 1
        let x = Expr::Var("x".to_string());
        let expr = x.clone() / x;
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(1));
    }

    #[test]
    fn test_simplify_div_zero_numerator() {
        // 0 / x = 0
        let expr = Expr::Const(0) / Expr::Var("x".to_string());
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(0));
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_simplify_div_zero_denominator() {
        // x / 0 should panic
        let expr = Expr::Var("x".to_string()) / Expr::Const(0);
        let _ = expr.simplify();
    }

    #[test]
    fn test_simplify_div_const() {
        // 10 / 2 = 5
        let expr = Expr::Const(10) / Expr::Const(2);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(5));
    }

    #[test]
    fn test_simplify_rem_one() {
        // x % 1 = 0
        let expr = Expr::Var("x".to_string()) % Expr::Const(1);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(0));
    }

    #[test]
    fn test_simplify_rem_self() {
        // x % x = 0
        let x = Expr::Var("x".to_string());
        let expr = x.clone() % x;
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(0));
    }

    #[test]
    fn test_simplify_rem_const() {
        // 10 % 3 = 1
        let expr = Expr::Const(10) % Expr::Const(3);
        let simplified = expr.simplify();
        assert_eq!(simplified, Expr::Const(1));
    }

    #[test]
    fn test_simplify_complex_expr() {
        // (x + 0) * 1 = x
        let x = Expr::Var("x".to_string());
        let expr = (x.clone() + 0) * 1;
        let simplified = expr.simplify();
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_display_const() {
        let expr = Expr::Const(42);
        assert_eq!(format!("{}", expr), "42");
    }

    #[test]
    fn test_display_var() {
        let expr = Expr::Var("x".to_string());
        assert_eq!(format!("{}", expr), "x");
    }

    #[test]
    fn test_display_add() {
        let expr = Expr::Const(2) + Expr::Const(3);
        assert_eq!(format!("{}", expr), "2 + 3");
    }

    #[test]
    fn test_display_sub() {
        let expr = Expr::Const(5) - Expr::Const(3);
        assert_eq!(format!("{}", expr), "5 - 3");
    }

    #[test]
    fn test_display_mul() {
        let expr = Expr::Const(4) * Expr::Const(5);
        assert_eq!(format!("{}", expr), "4 * 5");
    }

    #[test]
    fn test_display_complex() {
        // (2 + 3) * 4
        let expr = (Expr::Const(2) + Expr::Const(3)) * Expr::Const(4);
        assert_eq!(format!("{}", expr), "(2 + 3) * 4");
    }

    #[test]
    fn test_assign_operators() {
        let mut expr = Expr::Const(5);
        expr += 3;
        assert_eq!(
            expr,
            Expr::Add(Box::new(Expr::Const(5)), Box::new(Expr::Const(3)))
        );

        let mut expr = Expr::Const(10);
        expr -= 3;
        assert_eq!(
            expr,
            Expr::Sub(Box::new(Expr::Const(10)), Box::new(Expr::Const(3)))
        );

        let mut expr = Expr::Const(4);
        expr *= 5;
        assert_eq!(
            expr,
            Expr::Mul(Box::new(Expr::Const(4)), Box::new(Expr::Const(5)))
        );

        let mut expr = Expr::Const(20);
        expr /= 4;
        assert_eq!(
            expr,
            Expr::Div(Box::new(Expr::Const(20)), Box::new(Expr::Const(4)))
        );

        let mut expr = Expr::Const(10);
        expr %= 3;
        assert_eq!(
            expr,
            Expr::Rem(Box::new(Expr::Const(10)), Box::new(Expr::Const(3)))
        );
    }

    #[test]
    fn test_to_astnode_const() {
        use crate::ast::{AstNode, Literal};

        let expr = Expr::Const(42);
        let ast: AstNode = expr.into();

        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Const node with Isize(42)"),
        }
    }

    #[test]
    fn test_to_astnode_add() {
        use crate::ast::{AstNode, Literal};

        // 定数の加算をテスト
        let expr = Expr::Const(2) + Expr::Const(3);
        let ast: AstNode = expr.into();

        // After simplify, this becomes Const(5)
        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 5),
            _ => panic!("Expected Const(5) after simplification"),
        }

        // 変数の加算もサポートされるようになった
        let a = Expr::Var("a".to_string());
        let b = Expr::Var("b".to_string());
        let expr = Expr::Add(Box::new(a), Box::new(b));
        let ast: AstNode = expr.into();

        // 変数を含む加算はAdd nodeとして変換される
        match ast {
            AstNode::Add(left, right) => match (*left, *right) {
                (AstNode::Var(name_a), AstNode::Var(name_b)) => {
                    assert_eq!(name_a, "a");
                    assert_eq!(name_b, "b");
                }
                _ => panic!("Expected Var nodes"),
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_to_astnode_sub() {
        use crate::ast::{AstNode, Literal};

        // After simplify: 5 - 3 = 2
        let expr = Expr::Const(5) - Expr::Const(3);
        let ast: AstNode = expr.into();

        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 2),
            _ => panic!("Expected Const(2) after simplification"),
        }
    }

    #[test]
    fn test_to_astnode_mul() {
        use crate::ast::{AstNode, Literal};

        // After simplify: 4 * 5 = 20
        let expr = Expr::Const(4) * Expr::Const(5);
        let ast: AstNode = expr.into();

        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 20),
            _ => panic!("Expected Const(20) after simplification"),
        }
    }

    #[test]
    fn test_to_astnode_div() {
        use crate::ast::{AstNode, Literal};

        // After simplify: 10 / 2 = 5
        let expr = Expr::Const(10) / Expr::Const(2);
        let ast: AstNode = expr.into();

        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 5),
            _ => panic!("Expected Const(5) after simplification"),
        }
    }

    #[test]
    fn test_to_astnode_rem() {
        use crate::ast::{AstNode, Literal};

        // After simplify: 10 % 3 = 1
        let expr = Expr::Const(10) % Expr::Const(3);
        let ast: AstNode = expr.into();

        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 1),
            _ => panic!("Expected Const(1) after simplification"),
        }
    }

    #[test]
    fn test_to_astnode_complex() {
        use crate::ast::{AstNode, Literal};

        // After simplify: (2 + 3) * 4 = 5 * 4 = 20
        let expr = (Expr::Const(2) + Expr::Const(3)) * Expr::Const(4);
        let ast: AstNode = expr.into();

        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 20),
            _ => panic!("Expected Const(20) after simplification"),
        }
    }

    #[test]
    fn test_to_astnode_with_simplify() {
        use crate::ast::{AstNode, Literal};

        // 0 + 5 should simplify to 5 before conversion
        let expr = Expr::Const(0) + Expr::Const(5);
        let ast: AstNode = expr.into();

        match ast {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 5),
            _ => panic!("Expected simplified Const(5)"),
        }
    }

    #[test]
    fn test_to_astnode_var() {
        use crate::ast::AstNode;

        let expr = Expr::Var("x".to_string());
        let ast: AstNode = expr.into();

        // Varは正常に変換されるはず
        match ast {
            AstNode::Var(name) => assert_eq!(name, "x"),
            _ => panic!("Expected Var node"),
        }
    }
}
