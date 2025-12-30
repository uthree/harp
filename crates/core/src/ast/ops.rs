use super::{AstNode, Literal};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

// Operator overloading for AstNode with Into<AstNode> abstraction

impl<T: Into<AstNode>> Add<T> for AstNode {
    type Output = AstNode;

    fn add(self, rhs: T) -> AstNode {
        AstNode::Add(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<AstNode>> Mul<T> for AstNode {
    type Output = AstNode;

    fn mul(self, rhs: T) -> AstNode {
        AstNode::Mul(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<AstNode>> Rem<T> for AstNode {
    type Output = AstNode;

    fn rem(self, rhs: T) -> AstNode {
        AstNode::Rem(Box::new(self), Box::new(rhs.into()))
    }
}

// Subtraction: a - b = a + (-b)
impl<T: Into<AstNode>> Sub<T> for AstNode {
    type Output = AstNode;

    fn sub(self, rhs: T) -> AstNode {
        self + (-rhs.into())
    }
}

// Division: a / b = a * recip(b)
impl<T: Into<AstNode>> Div<T> for AstNode {
    type Output = AstNode;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: T) -> AstNode {
        // Division is implemented as multiplication by reciprocal: a / b = a * recip(b)
        self * AstNode::Recip(Box::new(rhs.into()))
    }
}

// Negation: -x = -1 * x
impl Neg for AstNode {
    type Output = AstNode;

    fn neg(self) -> AstNode {
        AstNode::Const(Literal::F32(-1.0)) * self
    }
}

// Bitwise operations
impl<T: Into<AstNode>> BitAnd<T> for AstNode {
    type Output = AstNode;

    fn bitand(self, rhs: T) -> AstNode {
        AstNode::BitwiseAnd(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<AstNode>> BitOr<T> for AstNode {
    type Output = AstNode;

    fn bitor(self, rhs: T) -> AstNode {
        AstNode::BitwiseOr(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<AstNode>> BitXor<T> for AstNode {
    type Output = AstNode;

    fn bitxor(self, rhs: T) -> AstNode {
        AstNode::BitwiseXor(Box::new(self), Box::new(rhs.into()))
    }
}

impl Not for AstNode {
    type Output = AstNode;

    fn not(self) -> AstNode {
        AstNode::BitwiseNot(Box::new(self))
    }
}

// Shift operations
impl<T: Into<AstNode>> Shl<T> for AstNode {
    type Output = AstNode;

    fn shl(self, rhs: T) -> AstNode {
        AstNode::LeftShift(Box::new(self), Box::new(rhs.into()))
    }
}

impl<T: Into<AstNode>> Shr<T> for AstNode {
    type Output = AstNode;

    fn shr(self, rhs: T) -> AstNode {
        AstNode::RightShift(Box::new(self), Box::new(rhs.into()))
    }
}

// Into<AstNode> implementations for numeric types
impl From<f32> for AstNode {
    fn from(value: f32) -> Self {
        AstNode::Const(Literal::F32(value))
    }
}

impl From<isize> for AstNode {
    fn from(value: isize) -> Self {
        AstNode::Const(Literal::I64(value as i64))
    }
}

impl From<usize> for AstNode {
    fn from(value: usize) -> Self {
        AstNode::Const(Literal::I64(value as i64))
    }
}

impl From<i32> for AstNode {
    fn from(value: i32) -> Self {
        AstNode::Const(Literal::I64(value as i64))
    }
}

impl From<i64> for AstNode {
    fn from(value: i64) -> Self {
        AstNode::Const(Literal::I64(value))
    }
}

// Reverse operations: numeric + AstNode
macro_rules! impl_reverse_ops {
    ($ty:ty) => {
        impl Add<AstNode> for $ty {
            type Output = AstNode;
            fn add(self, rhs: AstNode) -> AstNode {
                AstNode::from(self) + rhs
            }
        }

        impl Sub<AstNode> for $ty {
            type Output = AstNode;
            fn sub(self, rhs: AstNode) -> AstNode {
                AstNode::from(self) - rhs
            }
        }

        impl Mul<AstNode> for $ty {
            type Output = AstNode;
            fn mul(self, rhs: AstNode) -> AstNode {
                AstNode::from(self) * rhs
            }
        }

        impl Div<AstNode> for $ty {
            type Output = AstNode;
            fn div(self, rhs: AstNode) -> AstNode {
                AstNode::from(self) / rhs
            }
        }

        impl Rem<AstNode> for $ty {
            type Output = AstNode;
            fn rem(self, rhs: AstNode) -> AstNode {
                AstNode::from(self) % rhs
            }
        }
    };
}

impl_reverse_ops!(f32);
impl_reverse_ops!(isize);
impl_reverse_ops!(usize);
impl_reverse_ops!(i32);
impl_reverse_ops!(i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_operator() {
        let a = AstNode::from(1.0f32);
        let b = AstNode::from(2.0f32);
        let sum = a + b;

        match sum {
            AstNode::Add(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::F32(l)), AstNode::Const(Literal::F32(r))) => {
                    assert_eq!(l, 1.0);
                    assert_eq!(r, 2.0);
                }
                _ => panic!("Expected F32 constants in Add node"),
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_add_with_literal() {
        // Test Into<AstNode> abstraction with f32
        let a = AstNode::from(1.0f32);
        let sum = a + 2.0f32;

        match sum {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_mul_operator() {
        let a = AstNode::from(3.0f32);
        let b = AstNode::from(4.0f32);
        let product = a * b;

        match product {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_mul_with_literal() {
        let a = AstNode::from(3.0f32);
        let product = a * 4.0f32;

        match product {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_rem_operator() {
        let a = AstNode::from(5.0f32);
        let b = AstNode::from(2.0f32);
        let remainder = a % b;

        match remainder {
            AstNode::Rem(_, _) => {}
            _ => panic!("Expected Rem node"),
        }
    }

    #[test]
    fn test_sub_operator() {
        // a - b = a + (-b)
        let a = AstNode::from(5.0f32);
        let b = AstNode::from(3.0f32);
        let diff = a - b;

        // Should be Add(a, Mul(Const(-1), b))
        match diff {
            AstNode::Add(_, right) => match *right {
                AstNode::Mul(_, _) => {}
                _ => panic!("Expected Mul node for negation"),
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_sub_with_literal() {
        let a = AstNode::from(5.0f32);
        let diff = a - 3.0f32;

        match diff {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_div_operator() {
        // a / b = a * recip(b)
        let a = AstNode::from(10.0f32);
        let b = AstNode::from(2.0f32);
        let quotient = a / b;

        // Should be Mul(a, Recip(b))
        match quotient {
            AstNode::Mul(_, right) => match *right {
                AstNode::Recip(_) => {}
                _ => panic!("Expected Recip node"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_div_with_literal() {
        let a = AstNode::from(10.0f32);
        let quotient = a / 2.0f32;

        match quotient {
            AstNode::Mul(_, right) => match *right {
                AstNode::Recip(_) => {}
                _ => panic!("Expected Recip node"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_neg_operator() {
        // -x = -1 * x
        let a = AstNode::from(5.0f32);
        let neg_a = -a;

        match neg_a {
            AstNode::Mul(left, _) => match *left {
                AstNode::Const(Literal::F32(v)) => assert_eq!(v, -1.0),
                _ => panic!("Expected -1 constant"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_composite_operators() {
        // Test (a + b) * c
        let a = AstNode::from(1.0f32);
        let b = AstNode::from(2.0f32);
        let c = AstNode::from(3.0f32);
        let result = (a + b) * c;

        match result {
            AstNode::Mul(left, right) => match (*left, *right) {
                (AstNode::Add(_, _), AstNode::Const(Literal::F32(v))) => {
                    assert_eq!(v, 3.0);
                }
                _ => panic!("Expected Add node and F32 constant"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_complex_expression() {
        // Test (a - b) / (c + d)
        let a = AstNode::from(10.0f32);
        let b = AstNode::from(2.0f32);
        let c = AstNode::from(3.0f32);
        let d = AstNode::from(1.0f32);

        let result = (a - b) / (c + d);

        // Should be Mul(Add(...), Recip(Add(...)))
        match result {
            AstNode::Mul(_, right) => match *right {
                AstNode::Recip(inner) => match *inner {
                    AstNode::Add(_, _) => {}
                    _ => panic!("Expected Add inside Recip"),
                },
                _ => panic!("Expected Recip node"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_from_numeric_types() {
        let f32_node = AstNode::from(3.14f32);
        match f32_node {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 3.14),
            _ => panic!("Expected F32 constant"),
        }

        let isize_node = AstNode::from(42isize);
        match isize_node {
            AstNode::Const(Literal::I64(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Int constant"),
        }

        let usize_node = AstNode::from(100usize);
        match usize_node {
            AstNode::Const(Literal::I64(v)) => assert_eq!(v, 100),
            _ => panic!("Expected Int constant"),
        }
    }

    #[test]
    fn test_bitwise_and_operator() {
        let a = AstNode::from(5isize);
        let b = AstNode::from(3isize);
        let result = a & b;

        match result {
            AstNode::BitwiseAnd(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::I64(l)), AstNode::Const(Literal::I64(r))) => {
                    assert_eq!(l, 5);
                    assert_eq!(r, 3);
                }
                _ => panic!("Expected Int constants in BitwiseAnd node"),
            },
            _ => panic!("Expected BitwiseAnd node"),
        }
    }

    #[test]
    fn test_bitwise_or_operator() {
        let a = AstNode::from(5isize);
        let b = AstNode::from(3isize);
        let result = a | b;

        match result {
            AstNode::BitwiseOr(_, _) => {}
            _ => panic!("Expected BitwiseOr node"),
        }
    }

    #[test]
    fn test_bitwise_xor_operator() {
        let a = AstNode::from(5isize);
        let b = AstNode::from(3isize);
        let result = a ^ b;

        match result {
            AstNode::BitwiseXor(_, _) => {}
            _ => panic!("Expected BitwiseXor node"),
        }
    }

    #[test]
    fn test_bitwise_not_operator() {
        let a = AstNode::from(5isize);
        let result = !a;

        match result {
            AstNode::BitwiseNot(inner) => match *inner {
                AstNode::Const(Literal::I64(v)) => assert_eq!(v, 5),
                _ => panic!("Expected Int constant"),
            },
            _ => panic!("Expected BitwiseNot node"),
        }
    }

    #[test]
    fn test_left_shift_operator() {
        let a = AstNode::from(4isize);
        let b = AstNode::from(2isize);
        let result = a << b;

        match result {
            AstNode::LeftShift(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::I64(l)), AstNode::Const(Literal::I64(r))) => {
                    assert_eq!(l, 4);
                    assert_eq!(r, 2);
                }
                _ => panic!("Expected Int constants in LeftShift node"),
            },
            _ => panic!("Expected LeftShift node"),
        }
    }

    #[test]
    fn test_right_shift_operator() {
        let a = AstNode::from(16isize);
        let b = AstNode::from(2isize);
        let result = a >> b;

        match result {
            AstNode::RightShift(_, _) => {}
            _ => panic!("Expected RightShift node"),
        }
    }

    #[test]
    fn test_bitwise_with_literal() {
        // Test Into<AstNode> abstraction with isize
        let a = AstNode::from(5isize);
        let result = a & 3isize;

        match result {
            AstNode::BitwiseAnd(_, _) => {}
            _ => panic!("Expected BitwiseAnd node"),
        }
    }

    #[test]
    fn test_composite_bitwise_operations() {
        // Test (a & b) | c
        let a = AstNode::from(5isize);
        let b = AstNode::from(3isize);
        let c = AstNode::from(2isize);
        let result = (a & b) | c;

        match result {
            AstNode::BitwiseOr(left, right) => match (*left, *right) {
                (AstNode::BitwiseAnd(_, _), AstNode::Const(Literal::I64(v))) => {
                    assert_eq!(v, 2);
                }
                _ => panic!("Expected BitwiseAnd node and Int constant"),
            },
            _ => panic!("Expected BitwiseOr node"),
        }
    }

    #[test]
    fn test_reverse_add_operations() {
        use crate::ast::helper::var;

        // numeric + AstNode
        let result = 2.0f32 + var("x");
        match result {
            AstNode::Add(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::F32(v)), AstNode::Var(name)) => {
                    assert_eq!(v, 2.0);
                    assert_eq!(name, "x");
                }
                _ => panic!("Expected F32 constant and Var"),
            },
            _ => panic!("Expected Add node"),
        }

        let result = 3isize + var("y");
        match result {
            AstNode::Add(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::I64(v)), AstNode::Var(name)) => {
                    assert_eq!(v, 3);
                    assert_eq!(name, "y");
                }
                _ => panic!("Expected Int constant and Var"),
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_reverse_mul_operations() {
        use crate::ast::helper::var;

        let result = 2.0f32 * var("x");
        match result {
            AstNode::Mul(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::F32(v)), AstNode::Var(_)) => {
                    assert_eq!(v, 2.0);
                }
                _ => panic!("Expected F32 constant and Var"),
            },
            _ => panic!("Expected Mul node"),
        }

        // i32 also works
        let result = 10i32 * var("z");
        match result {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_mixed_numeric_operations() {
        use crate::ast::helper::var;

        // Complex expression: 2 * x + 3
        let x = var("x");
        let result = 2isize * x + 3isize;
        match result {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node"),
        }

        // 10 - x
        let x = var("x");
        let result = 10isize - x;
        match result {
            AstNode::Add(_, _) => {} // Sub is Add(a, Neg(b))
            _ => panic!("Expected Add node"),
        }

        // 1.0 / x
        let x = var("x");
        let result = 1.0f32 / x;
        match result {
            AstNode::Mul(_, right) => match *right {
                AstNode::Recip(_) => {}
                _ => panic!("Expected Recip node"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_from_i32_and_i64() {
        let i32_node = AstNode::from(42i32);
        match i32_node {
            AstNode::Const(Literal::I64(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Int constant"),
        }

        let i64_node = AstNode::from(100i64);
        match i64_node {
            AstNode::Const(Literal::I64(v)) => assert_eq!(v, 100),
            _ => panic!("Expected Int constant"),
        }
    }
}
