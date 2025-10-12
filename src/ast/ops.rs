use super::{AstNode, ConstLiteral};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

// Macro to implement From trait for numeric types to AstNode
macro_rules! impl_from_num_for_astnode {
    ($(($t:ty, $v: ident)),*) => {
        $(
            impl From<$t> for AstNode {
                fn from(n: $t) -> Self {
                    AstNode::Const(ConstLiteral::$v(n))
                }
            }
        )*
    };
}
impl_from_num_for_astnode!((usize, Usize), (isize, Isize), (f32, F32), (bool, Bool));

impl From<ConstLiteral> for AstNode {
    fn from(c: ConstLiteral) -> Self {
        AstNode::Const(c)
    }
}

// Macro to implement binary operators for AstNode
macro_rules! impl_astnode_binary_op {
    ($trait:ident, $fname:ident, $variant:ident) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: T) -> Self::Output {
                AstNode::$variant(Box::new(self), Box::new(rhs.into()))
            }
        }

        impl $trait<&AstNode> for &AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: &AstNode) -> Self::Output {
                AstNode::$variant(Box::new(self.clone()), Box::new(rhs.clone()))
            }
        }
    };
}

impl_astnode_binary_op!(Add, add, Add);
impl_astnode_binary_op!(Mul, mul, Mul);
impl_astnode_binary_op!(Rem, rem, Rem);
impl_astnode_binary_op!(BitAnd, bitand, BitAnd);
impl_astnode_binary_op!(BitOr, bitor, BitOr);
impl_astnode_binary_op!(BitXor, bitxor, BitXor);
impl_astnode_binary_op!(Shl, shl, Shl);
impl_astnode_binary_op!(Shr, shr, Shr);

// Subtraction: a - b = a + (-b)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Into<AstNode>> Sub<T> for AstNode {
    type Output = AstNode;
    fn sub(self, rhs: T) -> Self::Output {
        self + AstNode::Neg(Box::new(rhs.into()))
    }
}

// Division: a / b = a * (1/b)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Into<AstNode>> Div<T> for AstNode {
    type Output = AstNode;
    fn div(self, rhs: T) -> Self::Output {
        self * AstNode::Recip(Box::new(rhs.into()))
    }
}

// Macro to implement assignment operators for AstNode
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
impl_expr_assign_op!(BitAndAssign, bitand_assign, &);
impl_expr_assign_op!(BitOrAssign, bitor_assign, |);
impl_expr_assign_op!(BitXorAssign, bitxor_assign, ^);
impl_expr_assign_op!(ShlAssign, shl_assign, <<);
impl_expr_assign_op!(ShrAssign, shr_assign, >>);

impl Neg for AstNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        AstNode::Neg(Box::new(self))
    }
}

impl Not for AstNode {
    type Output = Self;
    fn not(self) -> Self::Output {
        AstNode::BitNot(Box::new(self))
    }
}

// Macro to implement unary operations for AstNode
macro_rules! impl_astnode_unary_op {
    ($fname:ident, $variant:ident) => {
        impl AstNode {
            pub fn $fname(self) -> Self {
                AstNode::$variant(Box::new(self))
            }
        }
    };
}

impl_astnode_unary_op!(recip, Recip);
impl_astnode_unary_op!(sin, Sin);
impl_astnode_unary_op!(sqrt, Sqrt);
impl_astnode_unary_op!(exp2, Exp2);
impl_astnode_unary_op!(log2, Log2);
