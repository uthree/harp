use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,   // float
    Usize, // size_t

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    F32(f32),
    Usize(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstOp {
    Const(Const),
    Cast(DType),

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct AstNode {
    op: AstOp,
    args: Vec<AstNode>,
    dtype: DType,
}

impl AstNode {
    pub fn _new(op: AstOp, args: Vec<AstNode>, dtype: DType) -> Self {
        Self { op, args, dtype }
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

impl_from_num_for_astnode!((usize, Usize));

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
        let a = AstNode::from(10usize);
        let b = AstNode::from(5usize);
        let result = op_fn(a.clone(), b.clone());

        let expected = AstNode::_new(op, vec![a, b], DType::Usize);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(AstOp::Add, |mut a: AstNode, b: AstNode| { a += b; a })]
    #[case(AstOp::Sub, |mut a: AstNode, b: AstNode| { a -= b; a })]
    #[case(AstOp::Mul, |mut a: AstNode, b: AstNode| { a *= b; a })]
    #[case(AstOp::Div, |mut a: AstNode, b: AstNode| { a /= b; a })]
    #[case(AstOp::Rem, |mut a: AstNode, b: AstNode| { a %= b; a })]
    fn test_assign_operations(#[case] op: AstOp, #[case] op_fn: fn(AstNode, AstNode) -> AstNode) {
        let a = AstNode::from(10usize);
        let b = AstNode::from(5usize);
        let result = op_fn(a.clone(), b.clone());

        let expected = AstNode::_new(op, vec![a, b], DType::Usize);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_ast_node() {
        let node = AstNode::from(10usize);
        let neg_node = -node.clone();

        let expected = AstNode::_new(AstOp::Neg, vec![node], DType::Usize);

        assert_eq!(neg_node, expected);
    }
}
