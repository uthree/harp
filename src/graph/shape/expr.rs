//! This module provides a symbolic expression (`Expr`) type for representing
//! tensor shapes and strides.
//!
//! `Expr` allows for the creation of shape computations that can be simplified
//! and manipulated before being concretized into actual values. This is crucial
//! for deferred shape calculation and optimization in the computation graph.

use crate::ast::{AstNode, DType};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

use log::debug;

/// A trait for converting a Rust type into its corresponding `DType` enum variant.
pub trait IntoDType {
    fn into_dtype() -> DType;
}

impl IntoDType for f32 {
    fn into_dtype() -> DType {
        DType::F32
    }
}
impl IntoDType for i64 {
    fn into_dtype() -> DType {
        DType::I64
    }
}
// Add other types as needed...

/// Represents a symbolic expression for tensor dimensions and strides.
///
/// This enum can represent constants, symbolic variables (like 'N' for batch size),
/// and arithmetic operations between them. It is used to define tensor shapes
/// that can be manipulated and simplified algebraically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// A constant integer value.
    Const(i64),
    /// A symbolic variable, represented by a string identifier.
    Var(String),
    /// The sum of two expressions.
    Add(Box<Expr>, Box<Expr>),
    /// The difference of two expressions.
    Sub(Box<Expr>, Box<Expr>),
    /// The product of two expressions.
    Mul(Box<Expr>, Box<Expr>),
    /// The quotient of two expressions.
    Div(Box<Expr>, Box<Expr>),
    /// The remainder of two expressions.
    Rem(Box<Expr>, Box<Expr>),
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Var(v) => write!(f, "{v}"),
            Expr::Add(l, r) => write!(f, "({l} + {r})"),
            Expr::Sub(l, r) => write!(f, "({l} - {r})"),
            Expr::Mul(l, r) => write!(f, "({l} * {r})"),
            Expr::Div(l, r) => write!(f, "({l} / {r})"),
            Expr::Rem(l, r) => write!(f, "({l} % {r})"),
        }
    }
}

impl Expr {
    /// Creates a new symbolic variable expression.
    ///
    /// # Arguments
    ///
    /// * `name` - The identifier for the variable (e.g., "N").
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_string())
    }

    /// Simplifies the expression algebraically.
    ///
    /// This method applies a set of rules to reduce the complexity of the expression,
    /// such as constant folding (e.g., `2 + 3` -> `5`) and eliminating identity
    /// operations (e.g., `x * 1` -> `x`).
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::graph::shape::expr::Expr;
    ///
    /// let n = Expr::var("N");
    /// // ((N + 0) * 1) simplifies to N
    /// let expr = (n.clone() + 0).simplify() * 1;
    /// assert_eq!(expr.simplify(), n);
    /// ```
    pub fn simplify(self) -> Self {
        let before = self.clone();
        let simplified = match self {
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
        };
        if before != simplified {
            // debug!("Simplified shape expression: {before} -> {simplified}");
        }
        simplified
    }

    /// Recursively traverses the expression and collects the names of all `Var` nodes.
    pub fn collect_variables(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            Expr::Var(name) => {
                vars.insert(name.clone());
            }
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Rem(l, r) => {
                l.collect_variables(vars);
                r.collect_variables(vars);
            }
            Expr::Const(_) => {}
        }
    }

    /// Evaluates the expression to a concrete `i64` value.
    ///
    /// All variables in the expression must be present in the `vars` map.
    ///
    /// # Arguments
    ///
    /// * `vars` - A map from variable names to their concrete `i64` values.
    ///
    /// # Panics
    ///
    /// Panics if a variable is found in the expression but not in the `vars` map.
    pub fn evaluate(&self, vars: &std::collections::HashMap<String, i64>) -> i64 {
        match self {
            Expr::Const(c) => *c,
            Expr::Var(v) => *vars
                .get(v)
                .unwrap_or_else(|| panic!("Variable '{v}' not found in evaluation context")),
            Expr::Add(l, r) => l.evaluate(vars) + r.evaluate(vars),
            Expr::Sub(l, r) => l.evaluate(vars) - r.evaluate(vars),
            Expr::Mul(l, r) => l.evaluate(vars) * r.evaluate(vars),
            Expr::Div(l, r) => l.evaluate(vars) / r.evaluate(vars),
            Expr::Rem(l, r) => l.evaluate(vars) % r.evaluate(vars),
        }
    }
}

impl From<i64> for Expr {
    fn from(val: i64) -> Self {
        Expr::Const(val)
    }
}

/// Implements `From<T>` for `Expr` for various numeric types.
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

/// Implements a binary operator for `Expr` where the RHS can be converted into `Expr`.
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

/// Implements a binary operator for `i64` where the RHS is an `Expr`.
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

/// Implements an assignment operator for `Expr`.
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
            Expr::Const(c) => AstNode::from(c as u64).cast(DType::USize),
            Expr::Var(s) => AstNode::var(&s).with_type(DType::USize),
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
            crate::ast::AstOp::Var(s) => Expr::Var(s),
            crate::ast::AstOp::Const(c) => match c {
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
    use crate::ast::AstOp;

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

        assert_eq!(ast.op, AstOp::Mul);
        assert_eq!(ast.dtype, DType::USize);

        let lhs = ast.src[0].clone();
        let rhs = ast.src[1].clone();

        assert_eq!(lhs.op, AstOp::Add);
        assert_eq!(lhs.dtype, DType::USize);

        // Check that the constant `2` is correctly converted and casted.
        assert_eq!(rhs.op, AstOp::Cast(DType::USize));
        if let AstOp::Const(c) = rhs.src[0].op {
            assert_eq!(c, crate::ast::Const::U64(2));
        } else {
            panic!("Expected a const node, found {:?}", rhs.src[0].op);
        }
    }

    #[test]
    #[allow(
        clippy::erasing_op,
        clippy::identity_op,
        clippy::op_ref,
        clippy::modulo_one
    )]
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
