use crate::ast::{AstNode, ConstLiteral};
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

impl From<Expr> for AstNode {
    fn from(expr: Expr) -> Self {
        // 変換前にsimplifyして可読性を向上
        let expr = expr.simplify();
        match expr {
            Expr::Const(c) => AstNode::Const(ConstLiteral::Isize(c)),
            Expr::Var(s) => AstNode::Var(s),
            Expr::Add(l, r) => AstNode::Add(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Sub(l, r) => AstNode::Add(
                Box::new((*l).into()),
                Box::new(AstNode::Neg(Box::new((*r).into()))),
            ),
            Expr::Mul(l, r) => AstNode::Mul(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Div(l, r) => AstNode::Add(
                Box::new((*l).into()),
                Box::new(AstNode::Recip(Box::new((*r).into()))),
            ),
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
    fn test_simplify_add() {
        let a = Expr::Var("a".to_string());
        assert_eq!((a.clone() + 0).simplify(), a.clone());
        assert_eq!((Expr::from(0) + a.clone()).simplify(), a.clone());
        assert_eq!((Expr::from(2) + 3).simplify(), Expr::from(5));
    }

    #[test]
    fn test_simplify_sub() {
        let a = Expr::Var("a".to_string());
        assert_eq!((a.clone() - 0).simplify(), a.clone());
        assert_eq!((a.clone() - a.clone()).simplify(), Expr::from(0));
        assert_eq!((Expr::from(5) - 2).simplify(), Expr::from(3));
        assert_eq!(((a.clone() + 2) - 2).simplify(), a.clone());
    }

    #[test]
    fn test_simplify_mul() {
        let a = Expr::Var("a".to_string());
        assert_eq!((a.clone() * 1).simplify(), a.clone());
        assert_eq!((Expr::from(1) * a.clone()).simplify(), a.clone());
        assert_eq!((a.clone() * 0).simplify(), Expr::from(0));
        assert_eq!((Expr::from(0) * a.clone()).simplify(), Expr::from(0));
        assert_eq!((Expr::from(2) * 3).simplify(), Expr::from(6));
    }

    #[test]
    fn test_simplify_div() {
        let a = Expr::Var("a".to_string());
        assert_eq!((a.clone() / 1).simplify(), a.clone());
        assert_eq!((a.clone() / a.clone()).simplify(), Expr::from(1));
        assert_eq!((Expr::from(0) / a).simplify(), Expr::from(0));
        assert_eq!((Expr::from(6) / 2).simplify(), Expr::from(3));
    }

    #[test]
    #[should_panic]
    fn test_simplify_div_by_zero() {
        (Expr::from(1) / 0).simplify();
    }

    #[test]
    fn test_simplify_rem() {
        let a = Expr::Var("a".to_string());
        assert_eq!((a.clone() % 1).simplify(), Expr::from(0));
        assert_eq!((a.clone() % a.clone()).simplify(), Expr::from(0));
        assert_eq!((Expr::from(0) % a).simplify(), Expr::from(0));
        assert_eq!((Expr::from(7) % 2).simplify(), Expr::from(1));
    }

    #[test]
    #[should_panic]
    fn test_simplify_rem_by_zero() {
        (Expr::from(1) % 0).simplify();
    }

    #[test]
    fn test_simplify_complex() {
        let a = Expr::Var("a".to_string());
        let b = Expr::Var("b".to_string());
        // ((a + 0) * 1 + b) / 1 = a + b
        let expr = ((a.clone() + 0) * 1 + b.clone()) / 1;
        assert_eq!(expr.simplify(), a + b);
    }

    #[test]
    fn test_s_macro() {
        use crate::s;

        // Test the s! macro with various combinations
        let shape1 = s![1, 2, 3];
        assert_eq!(shape1, vec![Expr::from(1), Expr::from(2), Expr::from(3)]);

        let shape2 = s!["a", "b"];
        assert_eq!(
            shape2,
            vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())]
        );

        let shape3 = s![1, "N", 2, "batch_size"];
        assert_eq!(
            shape3,
            vec![
                Expr::from(1),
                Expr::Var("N".to_string()),
                Expr::from(2),
                Expr::Var("batch_size".to_string())
            ]
        );

        let shape4 = s![];
        let empty: Vec<Expr> = vec![];
        assert_eq!(shape4, empty);
    }

    #[test]
    fn test_s_macro_with_graph() {
        use crate::{ast::DType, graph::Graph, s};

        let mut graph = Graph::new();

        // Use the s! macro to create shapes for graph input
        let input_node = graph.input(DType::F32, s![1, "batch_size", 3, "height", "width"]);

        // Verify the shape
        let expected_shape = vec![
            Expr::from(1),
            Expr::Var("batch_size".to_string()),
            Expr::from(3),
            Expr::Var("height".to_string()),
            Expr::Var("width".to_string()),
        ];
        assert_eq!(input_node.view.shape(), expected_shape);

        // Also test with pure constants and pure variables
        let const_input = graph.input(DType::F32, s![2, 3, 4]);
        assert_eq!(
            const_input.view.shape(),
            &[Expr::from(2), Expr::from(3), Expr::from(4)]
        );

        let var_input = graph.input(DType::F32, s!["N", "C", "H", "W"]);
        assert_eq!(
            var_input.view.shape(),
            &[
                Expr::Var("N".to_string()),
                Expr::Var("C".to_string()),
                Expr::Var("H".to_string()),
                Expr::Var("W".to_string())
            ]
        );
    }
}

/// Create a vector of `Expr` with a mix of constants and variables.
///
/// This macro allows you to easily create shape vectors by mixing integer literals
/// and string literals (which become variables).
///
/// # Examples
///
/// ```
/// use harp::{s, graph::shape::Expr};
///
/// // Pure constants
/// let shape1 = s![1, 2, 3];
/// assert_eq!(shape1, vec![Expr::from(1), Expr::from(2), Expr::from(3)]);
///
/// // Pure variables
/// let shape2 = s!["a", "b"];
/// assert_eq!(shape2, vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())]);
///
/// // Mixed constants and variables
/// let shape3 = s![1, "N", 2, "batch_size"];
/// assert_eq!(shape3, vec![
///     Expr::from(1),
///     Expr::Var("N".to_string()),
///     Expr::from(2),
///     Expr::Var("batch_size".to_string())
/// ]);
///
/// // Empty shape
/// let shape4 = s![];
/// let empty: Vec<Expr> = vec![];
/// assert_eq!(shape4, empty);
/// ```
#[macro_export]
macro_rules! s {
    // Base case: empty
    () => {
        Vec::<$crate::graph::shape::Expr>::new()
    };

    // One or more elements
    ($($item:expr),+ $(,)?) => {
        vec![$($crate::graph::shape::Expr::from($item)),+]
    };
}
