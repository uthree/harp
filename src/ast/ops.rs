use super::{AstNode, Literal};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

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

    fn div(self, rhs: T) -> AstNode {
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

// Into<AstNode> implementations for numeric types
impl From<f32> for AstNode {
    fn from(value: f32) -> Self {
        AstNode::Const(Literal::F32(value))
    }
}

impl From<isize> for AstNode {
    fn from(value: isize) -> Self {
        AstNode::Const(Literal::Isize(value))
    }
}

impl From<usize> for AstNode {
    fn from(value: usize) -> Self {
        AstNode::Const(Literal::Usize(value))
    }
}

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
    fn test_from_numeric_types() {
        let f32_node = AstNode::from(3.14f32);
        match f32_node {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 3.14),
            _ => panic!("Expected F32 constant"),
        }

        let isize_node = AstNode::from(42isize);
        match isize_node {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Isize constant"),
        }

        let usize_node = AstNode::from(100usize);
        match usize_node {
            AstNode::Const(Literal::Usize(v)) => assert_eq!(v, 100),
            _ => panic!("Expected Usize constant"),
        }
    }
}
