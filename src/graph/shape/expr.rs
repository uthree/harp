use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(isize),
    Var(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
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
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_string())
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
    }

    #[rstest]
    #[case(Expr::Const(5), hashmap!{}, 5)]
    #[case(Expr::var("x"), hashmap!{"x".to_string() => 10}, 10)]
    #[case(Expr::var("x") + Expr::Const(5), hashmap!{"x".to_string() => 10}, 15)]
    #[case((Expr::var("x") * Expr::var("y")) - Expr::Const(1), hashmap!{"x".to_string() => 3, "y".to_string() => 4}, 11)]
    #[case(Expr::Const(-5), hashmap!{}, -5)]
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
