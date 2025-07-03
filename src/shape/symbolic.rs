use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Index,
    Int(isize),
    Var(String),

    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    Neg(Box<Self>),
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Int(i) => write!(f, "{}", i),
            Expr::Var(s) => write!(f, "{}", s),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Rem(a, b) => write!(f, "({} % {})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
            Expr::Index => write!(f, "[IDX]"),
        }
    }
}

impl Expr {
    pub fn simplify(self) -> Self {
        match self {
            Expr::Int(_) | Expr::Var(_) => self,
            Expr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a, b) {
                    (Expr::Int(a), Expr::Int(b)) => Expr::Int(a + b),
                    (expr, Expr::Int(0)) | (Expr::Int(0), expr) => expr,
                    (a, b) => Expr::Add(Box::new(a), Box::new(b)),
                }
            }
            Expr::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a, b) {
                    (Expr::Int(a), Expr::Int(b)) => Expr::Int(a - b),
                    (expr, Expr::Int(0)) => expr,
                    (a, b) if a == b => Expr::Int(0),
                    (a, b) => Expr::Sub(Box::new(a), Box::new(b)),
                }
            }
            Expr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a, b) {
                    (Expr::Int(a), Expr::Int(b)) => Expr::Int(a * b),
                    (_, Expr::Int(0)) | (Expr::Int(0), _) => Expr::Int(0),
                    (expr, Expr::Int(1)) | (Expr::Int(1), expr) => expr,
                    (a, b) => Expr::Mul(Box::new(a), Box::new(b)),
                }
            }
            Expr::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a, b) {
                    (Expr::Int(a), Expr::Int(b)) if b != 0 => Expr::Int(a / b),
                    (Expr::Int(0), _) => Expr::Int(0),
                    (expr, Expr::Int(1)) => expr,
                    (a, b) if a == b => Expr::Int(1),
                    (a, b) => Expr::Div(Box::new(a), Box::new(b)),
                }
            }
            Expr::Rem(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a, b) {
                    (Expr::Int(a), Expr::Int(b)) if b != 0 => Expr::Int(a % b),
                    (Expr::Int(0), _) => Expr::Int(0),
                    (_, Expr::Int(1)) => Expr::Int(0),
                    (a, b) => Expr::Rem(Box::new(a), Box::new(b)),
                }
            }
            Expr::Neg(expr) => match expr.simplify() {
                Expr::Int(val) => Expr::Int(-val),
                Expr::Neg(inner_expr) => *inner_expr,
                e => Expr::Neg(Box::new(e)),
            },
            Expr::Index => Expr::Index,
        }
    }

    pub fn replace(&self, var: &String, val: &Expr) -> Expr {
        match self {
            Expr::Var(s) if s == var => val.clone(),
            Expr::Add(a, b) => Expr::Add(Box::new(a.replace(var, val)), Box::new(b.replace(var, val))),
            Expr::Sub(a, b) => Expr::Sub(Box::new(a.replace(var, val)), Box::new(b.replace(var, val))),
            Expr::Mul(a, b) => Expr::Mul(Box::new(a.replace(var, val)), Box::new(b.replace(var, val))),
            Expr::Div(a, b) => Expr::Div(Box::new(a.replace(var, val)), Box::new(b.replace(var, val))),
            Expr::Rem(a, b) => Expr::Rem(Box::new(a.replace(var, val)), Box::new(b.replace(var, val))),
            Expr::Neg(a) => Expr::Neg(Box::new(a.replace(var, val))),
            _ => self.clone(),
        }
    }
}

impl<T: Into<Expr>> Add<T> for Expr {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<Expr>> Sub<T> for Expr {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::Sub(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<Expr>> Mul<T> for Expr {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<Expr>> Div<T> for Expr {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::Div(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<Expr>> Rem<T> for Expr {
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        Self::Rem(Box::new(self), Box::new(rhs.into()))
    }
}

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::Neg(Box::new(self))
    }
}

macro_rules! impl_from_int_for_expr {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Expr {
                fn from(i: $t) -> Self {
                    Self::Int(i as isize)
                }
            }
        )*
    };
}

impl_from_int_for_expr!(i32, i64, usize, u32, u64);

impl From<isize> for Expr {
    fn from(i: isize) -> Self {
        Self::Int(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_simplify() {
        let expr = Expr::Int(1) + Expr::Int(2);
        assert_eq!(expr.simplify(), Expr::Int(3));

        let expr = Expr::Var("x".to_string()) + Expr::Int(0);
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));

        let expr = Expr::Int(0) + Expr::Var("x".to_string());
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));
    }

    #[test]
    fn test_sub_simplify() {
        let expr = Expr::Int(3) - Expr::Int(2);
        assert_eq!(expr.simplify(), Expr::Int(1));

        let expr = Expr::Var("x".to_string()) - Expr::Int(0);
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));

        let expr = Expr::Var("x".to_string()) - Expr::Var("x".to_string());
        assert_eq!(expr.simplify(), Expr::Int(0));
    }

    #[test]
    fn test_mul_simplify() {
        let expr = Expr::Int(2) * Expr::Int(3);
        assert_eq!(expr.simplify(), Expr::Int(6));

        let expr = Expr::Var("x".to_string()) * Expr::Int(1);
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));

        let expr = Expr::Int(1) * Expr::Var("x".to_string());
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));

        let expr = Expr::Var("x".to_string()) * Expr::Int(0);
        assert_eq!(expr.simplify(), Expr::Int(0));

        let expr = Expr::Int(0) * Expr::Var("x".to_string());
        assert_eq!(expr.simplify(), Expr::Int(0));
    }

    #[test]
    fn test_div_simplify() {
        let expr = Expr::Int(6) / Expr::Int(3);
        assert_eq!(expr.simplify(), Expr::Int(2));

        let expr = Expr::Var("x".to_string()) / Expr::Int(1);
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));

        let expr = Expr::Int(0) / Expr::Var("x".to_string());
        assert_eq!(expr.simplify(), Expr::Int(0));

        let expr = Expr::Var("x".to_string()) / Expr::Var("x".to_string());
        assert_eq!(expr.simplify(), Expr::Int(1));
    }

    #[test]
    fn test_rem_simplify() {
        let expr = Expr::Int(7) % Expr::Int(3);
        assert_eq!(expr.simplify(), Expr::Int(1));

        let expr = Expr::Var("x".to_string()) % Expr::Int(1);
        assert_eq!(expr.simplify(), Expr::Int(0));

        let expr = Expr::Int(0) % Expr::Var("x".to_string());
        assert_eq!(expr.simplify(), Expr::Int(0));
    }

    #[test]
    fn test_neg_simplify() {
        let expr = -Expr::Int(5);
        assert_eq!(expr.simplify(), Expr::Int(-5));

        let expr = -(-Expr::Var("x".to_string()));
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));
    }

    #[test]
    fn test_complex_simplify() {
        let expr = (Expr::Var("x".to_string()) + Expr::Int(0)) * Expr::Int(1);
        assert_eq!(expr.simplify(), Expr::Var("x".to_string()));

        let expr = (Expr::Var("y".to_string()) - Expr::Var("y".to_string())) + Expr::Int(5);
        assert_eq!(expr.simplify(), Expr::Int(5));

        let expr = (Expr::Int(10) * Expr::Var("z".to_string()))
            - (Expr::Var("z".to_string()) * Expr::Int(9));
        let expected = Expr::Sub(
            Box::new(Expr::Mul(
                Box::new(Expr::Int(10)),
                Box::new(Expr::Var("z".to_string())),
            )),
            Box::new(Expr::Mul(
                Box::new(Expr::Var("z".to_string())),
                Box::new(Expr::Int(9)),
            )),
        );
        assert_eq!(expr.simplify(), expected.simplify());
    }

    #[test]
    fn test_from_into() {
        let expr: Expr = 42.into();
        assert_eq!(expr, Expr::Int(42));

        let expr: Expr = 42i32.into();
        assert_eq!(expr, Expr::Int(42));

        let expr: Expr = 42i64.into();
        assert_eq!(expr, Expr::Int(42));

        let expr: Expr = 42usize.into();
        assert_eq!(expr, Expr::Int(42));

        let expr: Expr = 42u32.into();
        assert_eq!(expr, Expr::Int(42));

        let expr: Expr = 42u64.into();
        assert_eq!(expr, Expr::Int(42));
    }

    #[test]
    fn test_display() {
        let expr = Expr::Add(Box::new(Expr::Var("i".to_string())), Box::new(Expr::Int(1)));
        assert_eq!(expr.to_string(), "(i + 1)");

        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Sub(
                Box::new(Expr::Var("y".to_string())),
                Box::new(Expr::Int(2)),
            )),
        );
        assert_eq!(expr.to_string(), "(x * (y - 2))");

        let expr = Expr::Neg(Box::new(Expr::Var("z".to_string())));
        assert_eq!(expr.to_string(), "(-z)");
    }

    #[test]
    fn test_ops_with_into() {
        let x = Expr::Var("x".to_string());
        assert_eq!(
            x.clone() + 1,
            Expr::Add(Box::new(x.clone()), Box::new(Expr::Int(1)))
        );

        let x = Expr::Var("x".to_string());
        assert_eq!(
            x.clone() - 1,
            Expr::Sub(Box::new(x.clone()), Box::new(Expr::Int(1)))
        );

        let x = Expr::Var("x".to_string());
        assert_eq!(
            x.clone() * 1,
            Expr::Mul(Box::new(x.clone()), Box::new(Expr::Int(1)))
        );

        let x = Expr::Var("x".to_string());
        assert_eq!(
            x.clone() / 1,
            Expr::Div(Box::new(x.clone()), Box::new(Expr::Int(1)))
        );

        let x = Expr::Var("x".to_string());
        assert_eq!(
            x.clone() % 1,
            Expr::Rem(Box::new(x.clone()), Box::new(Expr::Int(1)))
        );
    }

    #[test]
    fn test_replace() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let expr = x.clone() + Expr::Int(1);
        let replaced_expr = expr.replace(&"x".to_string(), &y);
        assert_eq!(replaced_expr, y + Expr::Int(1));
    }
}
