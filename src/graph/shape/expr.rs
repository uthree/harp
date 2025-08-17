use crate::ast::{AstNode, DType};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Const(isize),
    Bool(bool),
    Var(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),

    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
    Lt(Box<Self>, Box<Self>),
    Eq(Box<Self>, Box<Self>),
    Gt(Box<Self>, Box<Self>),
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
            Expr::And(l, r) => write!(f, "({l} && {r})"),
            Expr::Or(l, r) => write!(f, "({l} || {r})"),
            Expr::Not(e) => write!(f, "!{e}"),
            Expr::Lt(l, r) => write!(f, "({l} < {r})"),
            Expr::Eq(l, r) => write!(f, "({l} == {r})"),
            Expr::Gt(l, r) => write!(f, "({l} > {r})"),
            Expr::Bool(b) => write!(f, "{b}"),
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

    pub fn to_ast(&self) -> AstNode {
        match self {
            Expr::Const(c) => AstNode::from(*c),
            Expr::Var(v) => AstNode::var(v, DType::Usize), // Assuming shape variables are always Usize
            Expr::Add(l, r) => l.to_ast() + r.to_ast(),
            Expr::Sub(l, r) => l.to_ast() - r.to_ast(),
            Expr::Mul(l, r) => l.to_ast() * r.to_ast(),
            Expr::Div(l, r) => l.to_ast() / r.to_ast(),
            Expr::Rem(l, r) => l.to_ast() % r.to_ast(),
            // Boolean/comparison ops are not directly convertible to arithmetic AstNodes
            // in the current setup. They are used for shape analysis, not code generation.
            _ => unimplemented!("Cannot convert expression {:?} to AstNode yet", self),
        }
    }
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_string())
    }

    pub const TRUE: Expr = Expr::Bool(true);
    pub const FALSE: Expr = Expr::Bool(false);

    pub fn lt(self, rhs: impl Into<Expr>) -> Self {
        Self::Lt(Box::new(self), Box::new(rhs.into()))
    }

    pub fn eq(self, rhs: impl Into<Expr>) -> Self {
        Self::Eq(Box::new(self), Box::new(rhs.into()))
    }

    pub fn gt(self, rhs: impl Into<Expr>) -> Self {
        Self::Gt(Box::new(self), Box::new(rhs.into()))
    }

    pub fn and(self, rhs: impl Into<Expr>) -> Self {
        Self::And(Box::new(self), Box::new(rhs.into()))
    }

    pub fn or(self, rhs: impl Into<Expr>) -> Self {
        Self::Or(Box::new(self), Box::new(rhs.into()))
    }

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
            Expr::And(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Bool(true), e) | (e, Expr::Bool(true)) => e,
                    (Expr::Bool(false), _) | (_, Expr::Bool(false)) => Expr::Bool(false),
                    (l, r) => l.and(r),
                }
            }
            Expr::Or(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Bool(true), _) | (_, Expr::Bool(true)) => Expr::Bool(true),
                    (Expr::Bool(false), e) | (e, Expr::Bool(false)) => e,
                    (l, r) => l.or(r),
                }
            }
            Expr::Not(e) => match e.simplify() {
                Expr::Bool(b) => Expr::Bool(!b),
                Expr::Not(inner) => *inner,
                e => !e,
            },
            Expr::Lt(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(l), Expr::Const(r)) => Expr::Bool(l < r),
                    (l, r) => l.lt(r),
                }
            }
            Expr::Eq(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(l), Expr::Const(r)) => Expr::Bool(l == r),
                    (l, r) if l == r => Expr::Bool(true),
                    (l, r) => l.eq(r),
                }
            }
            Expr::Gt(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(l), Expr::Const(r)) => Expr::Bool(l > r),
                    (l, r) => l.gt(r),
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
            | Expr::Rem(l, r)
            | Expr::And(l, r)
            | Expr::Or(l, r)
            | Expr::Lt(l, r)
            | Expr::Eq(l, r)
            | Expr::Gt(l, r) => {
                l.collect_variables(vars);
                r.collect_variables(vars);
            }
            Expr::Not(e) => {
                e.collect_variables(vars);
            }
            Expr::Const(_) | Expr::Bool(_) => {}
        }
    }

    pub fn evaluate(&self, vars: &std::collections::HashMap<String, isize>) -> isize {
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
            Expr::And(l, r) => ((l.evaluate(vars) != 0) && (r.evaluate(vars) != 0)) as isize,
            Expr::Or(l, r) => ((l.evaluate(vars) != 0) || (r.evaluate(vars) != 0)) as isize,
            Expr::Not(e) => (e.evaluate(vars) == 0) as isize,
            Expr::Lt(l, r) => (l.evaluate(vars) < r.evaluate(vars)) as isize,
            Expr::Eq(l, r) => (l.evaluate(vars) == r.evaluate(vars)) as isize,
            Expr::Gt(l, r) => (l.evaluate(vars) > r.evaluate(vars)) as isize,
            Expr::Bool(b) => *b as isize,
        }
    }
}

impl Not for Expr {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Not(Box::new(self))
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

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::collections::{HashMap, HashSet};

    // Helper macro for creating HashMaps in tests
    macro_rules! hashmap {
        ($( $key: expr => $val: expr ),* $(,)?) => {
            [$(($key, $val),)*].into_iter().collect::<::std::collections::HashMap<_, _>>()
        };
    }

    #[rstest]
    #[case(Expr::Const(1), "1")]
    #[case(Expr::Var("x".to_string()), "x")]
    #[case(Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(1))), "(x + 1)")]
    #[case(Expr::Sub(Box::new(Expr::Var("y".to_string())), Box::new(Expr::Const(2))), "(y - 2)")]
    #[case(Expr::Mul(Box::new(Expr::Var("a".to_string())), Box::new(Expr::Var("b".to_string()))), "(a * b)")]
    #[case(Expr::var("x").lt(1), "(x < 1)")]
    #[case(Expr::var("x").eq(Expr::var("y")), "(x == y)")]
    #[case(Expr::var("x").gt(10), "(x > 10)")]
    #[case(Expr::var("x").and(Expr::var("y")), "(x && y)")]
    #[case(Expr::var("x").or(Expr::var("y")), "(x || y)")]
    #[case(!Expr::var("x"), "!x")]
    #[case(Expr::TRUE, "true")]
    fn test_display(#[case] expr: Expr, #[case] expected: &str) {
        assert_eq!(expr.to_string(), expected);
    }

    #[rstest]
    // Add
    #[case(Expr::Const(1) + Expr::Const(2), Expr::Const(3))]
    #[case(Expr::var("x") + Expr::Const(0), Expr::var("x"))]
    #[case(Expr::Const(0) + Expr::var("x"), Expr::var("x"))]
    // Sub
    #[case(Expr::Const(3) - Expr::Const(1), Expr::Const(2))]
    #[case(Expr::var("x") - Expr::Const(0), Expr::var("x"))]
    #[case(Expr::var("x") - Expr::var("x"), Expr::Const(0))]
    #[case((Expr::var("x") + Expr::Const(1)) - Expr::Const(1), Expr::var("x"))]
    // Mul
    #[case(Expr::Const(2) * Expr::Const(3), Expr::Const(6))]
    #[case(Expr::var("x") * Expr::Const(1), Expr::var("x"))]
    #[case(Expr::Const(1) * Expr::var("x"), Expr::var("x"))]
    #[case(Expr::var("x") * Expr::Const(0), Expr::Const(0))]
    #[case(Expr::Const(0) * Expr::var("x"), Expr::Const(0))]
    #[case(Expr::var("x") * Expr::Const(-1), -Expr::var("x"))]
    // Div
    #[case(Expr::Const(6) / Expr::Const(2), Expr::Const(3))]
    #[case(Expr::var("x") / Expr::Const(1), Expr::var("x"))]
    #[case(Expr::var("x") / Expr::var("x"), Expr::Const(1))]
    #[case(Expr::Const(0) / Expr::var("x"), Expr::Const(0))]
    // Rem
    #[case(Expr::Const(7) % Expr::Const(3), Expr::Const(1))]
    #[case(Expr::var("x") % Expr::Const(1), Expr::Const(0))]
    #[case(Expr::var("x") % Expr::var("x"), Expr::Const(0))]
    #[case(Expr::Const(0) % Expr::var("x"), Expr::Const(0))]
    // Logical and Comparison
    #[case(Expr::TRUE.and(Expr::var("x")), Expr::var("x"))]
    #[case(Expr::var("x").and(Expr::TRUE), Expr::var("x"))]
    #[case(Expr::FALSE.and(Expr::var("x")), Expr::FALSE)]
    #[case(Expr::var("x").and(Expr::FALSE), Expr::FALSE)]
    #[case(Expr::TRUE.or(Expr::var("x")), Expr::TRUE)]
    #[case(Expr::var("x").or(Expr::TRUE), Expr::TRUE)]
    #[case(Expr::FALSE.or(Expr::var("x")), Expr::var("x"))]
    #[case(Expr::var("x").or(Expr::FALSE), Expr::var("x"))]
    #[case(!Expr::TRUE, Expr::FALSE)]
    #[case(!Expr::FALSE, Expr::TRUE)]
    #[case(!(!Expr::var("x")), Expr::var("x"))]
    #[case(Expr::Const(1).lt(2), Expr::TRUE)]
    #[case(Expr::Const(2).lt(1), Expr::FALSE)]
    #[case(Expr::Const(1).eq(1), Expr::TRUE)]
    #[case(Expr::Const(1).eq(2), Expr::FALSE)]
    #[case(Expr::var("x").eq(Expr::var("x")), Expr::TRUE)]
    #[case(Expr::Const(2).gt(1), Expr::TRUE)]
    #[case(Expr::Const(1).gt(2), Expr::FALSE)]
    // Nested
    #[case((Expr::var("x") + Expr::Const(1)) - Expr::Const(1), Expr::var("x"))]
    #[case(((Expr::var("x") * Expr::Const(2)) + Expr::Const(3)).simplify(), Expr::var("x") * Expr::Const(2) + Expr::Const(3))]
    fn test_simplify(#[case] expr: Expr, #[case] expected: Expr) {
        assert_eq!(expr.simplify(), expected);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_simplify_div_by_zero() {
        let _ = (Expr::var("x") / Expr::Const(0)).simplify();
    }

    #[test]
    fn test_collect_variables() {
        let expr = (Expr::var("a") + Expr::var("b")) * (Expr::var("a") + Expr::Const(1));
        let mut vars = HashSet::new();
        expr.collect_variables(&mut vars);
        let expected: HashSet<String> =
            ["a".to_string(), "b".to_string()].iter().cloned().collect();
        assert_eq!(vars, expected);

        let expr_logic = (Expr::var("x").gt(0)).and(Expr::var("y").lt(10));
        let mut vars_logic = HashSet::new();
        expr_logic.collect_variables(&mut vars_logic);
        let expected_logic: HashSet<String> =
            ["x".to_string(), "y".to_string()].iter().cloned().collect();
        assert_eq!(vars_logic, expected_logic);
    }

    #[rstest]
    #[case(Expr::Const(5), hashmap!{}, 5)]
    #[case(Expr::var("x"), hashmap!{"x".to_string() => 10}, 10)]
    #[case(Expr::var("x") + Expr::Const(5), hashmap!{"x".to_string() => 10}, 15)]
    #[case((Expr::var("x") * Expr::var("y")) - Expr::Const(1), hashmap!{"x".to_string() => 3, "y".to_string() => 4}, 11)]
    #[case(Expr::Const(-5), hashmap!{}, -5)]
    #[case(Expr::TRUE, hashmap!{}, 1)]
    #[case(Expr::FALSE, hashmap!{}, 0)]
    #[case(Expr::var("x").lt(10), hashmap!{"x".to_string() => 5}, 1)]
    #[case(Expr::var("x").lt(10), hashmap!{"x".to_string() => 15}, 0)]
    #[case(Expr::var("x").eq(10), hashmap!{"x".to_string() => 10}, 1)]
    #[case(Expr::var("x").eq(10), hashmap!{"x".to_string() => 15}, 0)]
    #[case(Expr::var("x").gt(10), hashmap!{"x".to_string() => 15}, 1)]
    #[case(Expr::var("x").gt(10), hashmap!{"x".to_string() => 5}, 0)]
    #[case(Expr::var("x").gt(0).and(Expr::var("y").lt(10)), hashmap!{"x".to_string() => 5, "y".to_string() => 5}, 1)]
    #[case(Expr::var("x").gt(0).and(Expr::var("y").lt(10)), hashmap!{"x".to_string() => -1, "y".to_string() => 5}, 0)]
    #[case(Expr::var("x").gt(0).or(Expr::var("y").lt(10)), hashmap!{"x".to_string() => -1, "y".to_string() => 5}, 1)]
    #[case(Expr::var("x").gt(0).or(Expr::var("y").lt(10)), hashmap!{"x".to_string() => -1, "y".to_string() => 15}, 0)]
    #[case((!Expr::var("x").gt(0)), hashmap!{"x".to_string() => 5}, 0)]
    #[case((!Expr::var("x").gt(0)), hashmap!{"x".to_string() => -5}, 1)]
    fn test_evaluate(
        #[case] expr: Expr,
        #[case] context: HashMap<String, isize>,
        #[case] expected: isize,
    ) {
        assert_eq!(expr.evaluate(&context), expected);
    }

    #[test]
    #[should_panic(expected = "Variable 'z' not found in evaluation context")]
    fn test_evaluate_panic() {
        let expr = Expr::var("z");
        let context = HashMap::new();
        expr.evaluate(&context);
    }

    #[test]
    fn test_add_assign() {
        let mut expr = Expr::Const(1);
        expr += Expr::Const(2);
        assert_eq!(expr, Expr::Const(1) + Expr::Const(2));
    }

    #[test]
    fn test_sub_assign() {
        let mut expr = Expr::var("x");
        expr -= Expr::Const(1);
        assert_eq!(expr, Expr::var("x") - Expr::Const(1));
    }

    #[test]
    fn test_mul_assign() {
        let mut expr = Expr::Const(10);
        expr *= Expr::var("y");
        assert_eq!(expr, Expr::Const(10) * Expr::var("y"));
    }

    #[test]
    fn test_div_assign() {
        let mut expr = Expr::var("a");
        expr /= Expr::var("b");
        assert_eq!(expr, Expr::var("a") / Expr::var("b"));
    }

    #[test]
    fn test_rem_assign() {
        let mut expr = Expr::Const(5);
        expr %= Expr::Const(2);
        assert_eq!(expr, Expr::Const(5) % Expr::Const(2));
    }

    #[test]
    fn test_from_integer_conversion() {
        assert_eq!(Expr::from(10u8), Expr::Const(10));
        assert_eq!(Expr::from(100u16), Expr::Const(100));
        assert_eq!(Expr::from(1000u32), Expr::Const(1000));
        assert_eq!(Expr::from(10000u64), Expr::Const(10000));
        assert_eq!(Expr::from(1usize), Expr::Const(1));
        assert_eq!(Expr::from(10i8), Expr::Const(10));
        assert_eq!(Expr::from(100i16), Expr::Const(100));
        assert_eq!(Expr::from(1000i32), Expr::Const(1000));
        assert_eq!(Expr::from(10000i64), Expr::Const(10000));
        assert_eq!(Expr::from(1isize), Expr::Const(1));
        assert_eq!(Expr::from(-1i8), Expr::Const(-1));
    }

    #[test]
    fn test_mixed_type_operations() {
        let expr = Expr::var("x") + 10;
        assert_eq!(
            expr,
            Expr::Add(Box::new(Expr::var("x")), Box::new(Expr::Const(10)))
        );

        let expr2 = Expr::from(20) * Expr::var("y");
        assert_eq!(
            expr2,
            Expr::Mul(Box::new(Expr::Const(20)), Box::new(Expr::var("y")))
        );
    }

    #[test]
    fn test_neg() {
        let x = Expr::var("x");
        assert_eq!(-x.clone(), Expr::from(0isize) - x.clone());
    }

    #[test]
    fn test_double_neg_simplification() {
        let x = Expr::var("x");
        // The simplification for --x is handled inside the Neg trait which calls sub,
        // so we call simplify() to test the result.
        assert_eq!((-(-x.clone())).simplify(), x.clone());
    }

    #[test]
    fn test_neg_const_simplification() {
        let five = Expr::from(5);
        let neg_five = -five;
        assert_eq!(neg_five.simplify(), Expr::Const(-5));
    }
}
