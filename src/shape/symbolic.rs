use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Represents a symbolic expression for shape dimensions.
///
/// This enum allows for defining dimensions using integers, variables, or
/// arithmetic operations between them. It's used for symbolic shape tracking
/// and optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// Represents a subscript or an iteration variable. Its name is determined at compile time.
    Index,
    /// Represents a symbolic variable, identified by a string name.
    Var(String),
    /// Represents an integer literal.
    Int(isize),

    /// Addition of two expressions.
    Add(Box<Self>, Box<Self>),
    /// Subtraction of two expressions.
    Sub(Box<Self>, Box<Self>),
    /// Multiplication of two expressions.
    Mul(Box<Self>, Box<Self>),
    /// Division of two expressions.
    Div(Box<Self>, Box<Self>),
    /// Remainder of two expressions.
    Rem(Box<Self>, Box<Self>),
    /// Negation of an expression.
    Neg(Box<Self>),
}

impl Expr {
    /// Simplifies the symbolic expression.
    ///
    /// This method performs constant folding and basic algebraic simplifications
    /// to reduce the complexity of the expression.
    ///
    /// # Returns
    ///
    /// A new `Expr` representing the simplified form.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::shape::symbolic::Expr;
    ///
    /// let expr = Expr::Add(Box::new(Expr::Int(2)), Box::new(Expr::Int(3)));
    /// assert_eq!(expr.simplify(), Expr::Int(5));
    ///
    /// let expr = Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Int(0)));
    /// assert_eq!(expr.simplify(), Expr::Var("x".to_string()));
    /// ```
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
                        // If Index is on the right, swap to the left for canonical form.
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
                        // If Index is on the right, swap to the left for canonical form.
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

    /// Replaces all occurrences of a specific sub-expression with another expression.
    ///
    /// This is useful for substituting variables or indices with concrete values
    /// or other symbolic expressions.
    ///
    /// # Arguments
    ///
    /// * `old_expr` - The expression to be replaced.
    /// * `new_expr` - The expression to replace `old_expr` with.
    ///
    /// # Returns
    ///
    /// A new `Expr` with the replacements applied.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::shape::symbolic::Expr;
    ///
    /// let expr = Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Int(5)));
    /// let replaced = expr.replace(&Expr::Var("x".to_string()), &Expr::Int(10));
    /// assert_eq!(replaced, Expr::Add(Box::new(Expr::Int(10)), Box::new(Expr::Int(5))));
    /// ```
    pub fn replace(self, old_expr: &Expr, new_expr: &Expr) -> Expr {
        if &self == old_expr {
            return new_expr.clone();
        }

        match self {
            Expr::Add(a, b) => Expr::Add(
                Box::new(a.replace(old_expr, new_expr)),
                Box::new(b.replace(old_expr, new_expr)),
            ),
            Expr::Sub(a, b) => Expr::Sub(
                Box::new(a.replace(old_expr, new_expr)),
                Box::new(b.replace(old_expr, new_expr)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(a.replace(old_expr, new_expr)),
                Box::new(b.replace(old_expr, new_expr)),
            ),
            Expr::Div(a, b) => Expr::Div(
                Box::new(a.replace(old_expr, new_expr)),
                Box::new(b.replace(old_expr, new_expr)),
            ),
            Expr::Rem(a, b) => Expr::Rem(
                Box::new(a.replace(old_expr, new_expr)),
                Box::new(b.replace(old_expr, new_expr)),
            ),
            Expr::Neg(a) => Expr::Neg(Box::new(a.replace(old_expr, new_expr))),
            Expr::Index => self,
            Expr::Var(_) => self,
            Expr::Int(_) => self,
        }
    }
}

impl fmt::Display for Expr {
    /// Formats the `Expr` for display.
    ///
    /// This implementation provides a human-readable string representation of the symbolic expression.
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
        /// Implements an arithmetic operator for `Expr`.
        ///
        /// This macro reduces boilerplate for implementing `Add`, `Sub`, `Mul`, `Div`, and `Rem`
        /// traits for the `Expr` enum, allowing for natural arithmetic syntax.
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

    /// Implements the unary negation operator for `Expr`.
    fn neg(self) -> Self::Output {
        Self::Neg(Box::new(self))
    }
}

macro_rules! impl_from_int_for_expr {
    ($($t:ty),*) => {
        $(
            /// Implements `From` trait for various integer types to `Expr`.
            ///
            /// This allows direct conversion of integer literals into `Expr::Int`.
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
    /// Converts a string slice into an `Expr::Var`.
    ///
    /// This allows using string literals directly as symbolic variables.
    fn from(value: &str) -> Self {
        Expr::Var(value.to_string())
    }
}

impl From<String> for Expr {
    /// Converts a `String` into an `Expr::Var`.
    fn from(value: String) -> Self {
        Expr::Var(value)
    }
}
