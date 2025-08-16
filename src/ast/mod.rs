pub mod pattern;

use crate::graph::shape::expr::Expr as ShapeExpr;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,   // float
    Usize, // size_t
    Isize, // ssize_t

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array (for SIMD vectorization)

    Any, // for pattern matching
}

impl DType {
    pub fn zero(&self) -> AstNode {
        match self {
            DType::F32 => AstNode::from(0.0f32),
            DType::Usize => AstNode::from(0usize),
            DType::Isize => AstNode::from(0isize),
            _ => panic!("Cannot create a zero for non-numeric type {:?}", self),
        }
    }

    pub fn one(&self) -> AstNode {
        match self {
            DType::F32 => AstNode::from(1.0f32),
            DType::Usize => AstNode::from(1usize),
            DType::Isize => AstNode::from(1isize),
            _ => panic!("Cannot create a one for non-numeric type {:?}", self),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    F32(f32),
    Usize(usize),
    Isize(isize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstOp {
    Const(Const),
    Cast(DType),
    Var(String),

    Add,
    Mul,
    Sub,
    Div,
    Rem,
    Max,
    Sin,
    Exp2,
    Log2,
    Sqrt,
    Neg,

    // Logical and comparison
    And,
    Or,
    Not,
    Lt,
    Eq,
    Gt,

    Loop { counter: String },

    // for pattern matching
    Capture(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AstNode {
    pub op: AstOp,
    pub src: Vec<AstNode>,
    pub dtype: DType,
}

impl AstNode {
    pub fn _new(op: AstOp, src: Vec<AstNode>, dtype: DType) -> Self {
        Self { op, src, dtype }
    }

    pub fn capture(pos: usize) -> Self {
        AstNode::_new(AstOp::Capture(pos), vec![], DType::Any)
    }

    pub fn var(name: &str) -> Self {
        AstNode::_new(AstOp::Var(name.to_string()), vec![], DType::Any)
    }

    /// Matches the node against a pattern.
    ///
    /// If the node matches the pattern, it returns a vector of captured nodes.
    /// Otherwise, it returns `None`.
    pub fn matches(&self, pattern: &AstNode) -> Option<Vec<AstNode>> {
        let mut captures = Vec::new();
        if !self.matches_inner(pattern, &mut captures) {
            return None;
        }

        // The `astpat!` macro generates capture indices starting from 0.
        // We can rely on this to correctly size the results vector.
        // Find the maximum capture index to determine the size of the vector.
        let num_captures = captures.iter().map(|(i, _)| i).max().map_or(0, |m| m + 1);
        let mut result = vec![AstNode::from(0isize); num_captures];
        for (i, node) in captures {
            if i < num_captures {
                result[i] = node;
            }
        }
        Some(result)
    }

    fn matches_inner(&self, pattern: &AstNode, captures: &mut Vec<(usize, AstNode)>) -> bool {
        if let AstOp::Capture(pos) = &pattern.op {
            captures.push((*pos, self.clone()));
            return true;
        }

        if pattern.dtype != DType::Any && self.dtype != pattern.dtype {
            return false;
        }

        if self.op != pattern.op || self.src.len() != pattern.src.len() {
            return false;
        }

        for (arg, pattern_arg) in self.src.iter().zip(&pattern.src) {
            if !arg.matches_inner(pattern_arg, captures) {
                return false;
            }
        }

        true
    }
}

impl From<ShapeExpr> for AstNode {
    fn from(expr: ShapeExpr) -> Self {
        let dtype = DType::Isize;
        match expr {
            ShapeExpr::Const(c) => AstNode::_new(AstOp::Const(Const::Isize(c)), vec![], dtype),
            ShapeExpr::Var(v) => AstNode::_new(AstOp::Var(v), vec![], dtype),
            ShapeExpr::Add(l, r) => {
                AstNode::_new(AstOp::Add, vec![(*l).into(), (*r).into()], dtype)
            }
            ShapeExpr::Sub(l, r) => {
                AstNode::_new(AstOp::Sub, vec![(*l).into(), (*r).into()], dtype)
            }
            ShapeExpr::Mul(l, r) => {
                AstNode::_new(AstOp::Mul, vec![(*l).into(), (*r).into()], dtype)
            }
            ShapeExpr::Div(l, r) => {
                AstNode::_new(AstOp::Div, vec![(*l).into(), (*r).into()], dtype)
            }
            ShapeExpr::Rem(l, r) => {
                AstNode::_new(AstOp::Rem, vec![(*l).into(), (*r).into()], dtype)
            }
            ShapeExpr::Bool(b) => {
                AstNode::_new(AstOp::Const(Const::Isize(b as isize)), vec![], dtype)
            }
            ShapeExpr::And(l, r) => {
                AstNode::_new(AstOp::And, vec![(*l).into(), (*r).into()], dtype)
            }
            ShapeExpr::Or(l, r) => AstNode::_new(AstOp::Or, vec![(*l).into(), (*r).into()], dtype),
            ShapeExpr::Not(e) => AstNode::_new(AstOp::Not, vec![(*e).into()], dtype),
            ShapeExpr::Lt(l, r) => AstNode::_new(AstOp::Lt, vec![(*l).into(), (*r).into()], dtype),
            ShapeExpr::Eq(l, r) => AstNode::_new(AstOp::Eq, vec![(*l).into(), (*r).into()], dtype),
            ShapeExpr::Gt(l, r) => AstNode::_new(AstOp::Gt, vec![(*l).into(), (*r).into()], dtype),
        }
    }
}

macro_rules! impl_from_num_for_astnode {
    ($(($t:ty, $v: ident)),*) => {
        $(
            impl From<$t> for AstNode {
                fn from(n: $t) -> Self {
                    AstNode::_new(AstOp::Const(Const::$v(n)), vec![], DType::$v)
                }
            }
        )*
    };
}

impl_from_num_for_astnode!((f32, F32), (usize, Usize), (isize, Isize));

macro_rules! impl_astnode_binary_op {
    ($trait:ident, $fname:ident, $variant:ident) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: T) -> Self::Output {
                let dtype = (&self.dtype).clone();
                AstNode::_new(AstOp::$variant, vec![self, rhs.into()], dtype)
            }
        }
    };
}

impl_astnode_binary_op!(Add, add, Add);
impl_astnode_binary_op!(Mul, mul, Mul);
impl_astnode_binary_op!(Rem, rem, Rem);
impl_astnode_binary_op!(Sub, sub, Sub);
impl_astnode_binary_op!(Div, div, Div);

macro_rules! impl_expr_assign_op {
    ($trait:ident, $fname:ident, $op:tt) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
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

impl Neg for AstNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let dtype = self.dtype.clone();
        AstNode::_new(AstOp::Neg, vec![self], dtype)
    }
}

impl AstNode {
    pub fn and(self, rhs: impl Into<AstNode>) -> Self {
        let dtype = self.dtype.clone();
        AstNode::_new(AstOp::And, vec![self, rhs.into()], dtype)
    }

    pub fn or(self, rhs: impl Into<AstNode>) -> Self {
        let dtype = self.dtype.clone();
        AstNode::_new(AstOp::Or, vec![self, rhs.into()], dtype)
    }

    pub fn not(self) -> Self {
        let dtype = self.dtype.clone();
        AstNode::_new(AstOp::Not, vec![self], dtype)
    }

    pub fn lt(self, rhs: impl Into<AstNode>) -> Self {
        let dtype = self.dtype.clone();
        AstNode::_new(AstOp::Lt, vec![self, rhs.into()], dtype)
    }

    pub fn eq(self, rhs: impl Into<AstNode>) -> Self {
        let dtype = self.dtype.clone();
        AstNode::_new(AstOp::Eq, vec![self, rhs.into()], dtype)
    }

    pub fn gt(self, rhs: impl Into<AstNode>) -> Self {
        let dtype = self.dtype.clone();
        AstNode::_new(AstOp::Gt, vec![self, rhs.into()], dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_from_usize() {
        let node = AstNode::from(42usize);
        let expected = AstNode::_new(AstOp::Const(Const::Usize(42)), vec![], DType::Usize);
        assert_eq!(node, expected);
    }

    #[rstest]
    #[case(AstOp::Add, |a, b| a + b)]
    #[case(AstOp::Sub, |a, b| a - b)]
    #[case(AstOp::Mul, |a, b| a * b)]
    #[case(AstOp::Div, |a, b| a / b)]
    #[case(AstOp::Rem, |a, b| a % b)]
    fn test_binary_operations(#[case] op: AstOp, #[case] op_fn: fn(AstNode, AstNode) -> AstNode) {
        let a = AstNode::from(10isize);
        let b = AstNode::from(5isize);
        let result = op_fn(a.clone(), b.clone());

        let expected = AstNode::_new(op, vec![a, b], DType::Isize);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(AstOp::Add, |mut a: AstNode, b: AstNode| { a += b; a })]
    #[case(AstOp::Sub, |mut a: AstNode, b: AstNode| { a -= b; a })]
    #[case(AstOp::Mul, |mut a: AstNode, b: AstNode| { a *= b; a })]
    #[case(AstOp::Div, |mut a: AstNode, b: AstNode| { a /= b; a })]
    #[case(AstOp::Rem, |mut a: AstNode, b: AstNode| { a %= b; a })]
    fn test_assign_operations(#[case] op: AstOp, #[case] op_fn: fn(AstNode, AstNode) -> AstNode) {
        let a = AstNode::from(10isize);
        let b = AstNode::from(5isize);
        let result = op_fn(a.clone(), b.clone());

        let expected = AstNode::_new(op, vec![a, b], DType::Isize);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_ast_node() {
        let node = AstNode::from(10isize);
        let neg_node = -node.clone();

        let expected = AstNode::_new(AstOp::Neg, vec![node], DType::Isize);

        assert_eq!(neg_node, expected);
    }

    #[test]
    fn test_from_shape_expr() {
        let shape_expr = ShapeExpr::Add(
            Box::new(ShapeExpr::Var("a".to_string())),
            Box::new(ShapeExpr::Const(1)),
        );
        let ast_node: AstNode = shape_expr.into();

        let expected_ast = AstNode::_new(
            AstOp::Add,
            vec![
                AstNode::_new(AstOp::Var("a".to_string()), vec![], DType::Isize),
                AstNode::_new(AstOp::Const(Const::Isize(1)), vec![], DType::Isize),
            ],
            DType::Isize,
        );

        assert_eq!(ast_node, expected_ast);
    }

    #[cfg(test)]
    mod dtype_tests {
        use super::*;

        #[test]
        fn test_dtype_zero() {
            assert_eq!(DType::F32.zero(), AstNode::from(0.0f32));
            assert_eq!(DType::Usize.zero(), AstNode::from(0usize));
            assert_eq!(DType::Isize.zero(), AstNode::from(0isize));
        }

        #[test]
        fn test_dtype_one() {
            assert_eq!(DType::F32.one(), AstNode::from(1.0f32));
            assert_eq!(DType::Usize.one(), AstNode::from(1usize));
            assert_eq!(DType::Isize.one(), AstNode::from(1isize));
        }

        #[test]
        #[should_panic]
        fn test_dtype_zero_panic() {
            DType::Any.zero();
        }

        #[test]
        #[should_panic]
        fn test_dtype_one_panic() {
            DType::Ptr(Box::new(DType::F32)).one();
        }
    }
}
