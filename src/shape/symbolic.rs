use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Index, // This represents a subscript, and its name is determined at compile time.
    Var(String),
    Int(isize),

    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    Neg(Box<Self>),
}

impl Expr {
    pub fn simplify(self) -> Self {
        match self {
            Expr::Int(_) => self,
            Expr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (a, b) {
                    (Expr::Int(a), Expr::Int(b)) => Expr::Int(a + b),
                    (expr, Expr::Int(0)) | (Expr::Int(0), expr) => expr,
                    (a, b) => {
                        // If Index is on the right, swap to the left
                        if matches!(b, Expr::Index) && !matches!(a, Expr::Index) {
                            Expr::Add(Box::new(b), Box::new(a))
                        } else {
                            Expr::Add(Box::new(a), Box::new(b))
                        }
                    }
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
                    (a, b) => {
                        // If Index is on the right, swap to the left
                        if matches!(b, Expr::Index) && !matches!(a, Expr::Index) {
                            Expr::Mul(Box::new(b), Box::new(a))
                        } else {
                            Expr::Mul(Box::new(a), Box::new(b))
                        }
                    }
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
            Expr::Var(v) => Expr::Var(v),
        }
    }

    pub fn replace(self, old_expr: &Expr, new_expr: &Expr) -> Expr {
        if &self == old_expr {
            return new_expr.clone();
        }

        match self {
            Expr::Add(a, b) => {
                Expr::Add(Box::new(a.replace(old_expr, new_expr)), Box::new(b.replace(old_expr, new_expr)))
            }
            Expr::Sub(a, b) => {
                Expr::Sub(Box::new(a.replace(old_expr, new_expr)), Box::new(b.replace(old_expr, new_expr)))
            }
            Expr::Mul(a, b) => {
                Expr::Mul(Box::new(a.replace(old_expr, new_expr)), Box::new(b.replace(old_expr, new_expr)))
            }
            Expr::Div(a, b) => {
                Expr::Div(Box::new(a.replace(old_expr, new_expr)), Box::new(b.replace(old_expr, new_expr)))
            }
            Expr::Rem(a, b) => {
                Expr::Rem(Box::new(a.replace(old_expr, new_expr)), Box::new(b.replace(old_expr, new_expr)))
            }
            Expr::Neg(a) => Expr::Neg(Box::new(a.replace(old_expr, new_expr))),
            Expr::Index => self,
            Expr::Var(_) => self,
            Expr::Int(_) => self,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Index => write!(f, "idx"),
            Expr::Var(s) => write!(f, "{}", s),
            Expr::Int(i) => write!(f, "{}", i),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Rem(a, b) => write!(f, "({} % {})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
        }
    }
}

macro_rules! impl_expr_op {
    ($trait:ident, $func:ident, $variant:ident) => {
        impl<T: Into<Expr>> $trait<T> for Expr {
            type Output = Self;

            fn $func(self, rhs: T) -> Self::Output {
                Self::$variant(Box::new(self), Box::new(rhs.into()))
            }
        }
    };
}

impl_expr_op!(Add, add, Add);
impl_expr_op!(Sub, sub, Sub);
impl_expr_op!(Mul, mul, Mul);
impl_expr_op!(Div, div, Div);
impl_expr_op!(Rem, rem, Rem);

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

impl_from_int_for_expr!(i32, i64, usize, u32, u64, isize);

impl From<&str> for Expr {
    fn from(value: &str) -> Self {
        Expr::Var(value.to_string())
    }
}

impl From<String> for Expr {
    fn from(value: String) -> Self {
        Expr::Var(value)
    }
}
