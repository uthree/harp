use super::AstNode;
use std::ops::{Add, Mul, Rem};

// Operator overloading for AstNode

impl Add for AstNode {
    type Output = AstNode;

    fn add(self, rhs: AstNode) -> AstNode {
        AstNode::Add(Box::new(self), Box::new(rhs))
    }
}

impl Mul for AstNode {
    type Output = AstNode;

    fn mul(self, rhs: AstNode) -> AstNode {
        AstNode::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Rem for AstNode {
    type Output = AstNode;

    fn rem(self, rhs: AstNode) -> AstNode {
        AstNode::Rem(Box::new(self), Box::new(rhs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_add_operator() {
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
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
    fn test_mul_operator() {
        let a = AstNode::Const(3.0f32.into());
        let b = AstNode::Const(4.0f32.into());
        let product = a * b;

        match product {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_rem_operator() {
        let a = AstNode::Const(5.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let remainder = a % b;

        match remainder {
            AstNode::Rem(_, _) => {}
            _ => panic!("Expected Rem node"),
        }
    }

    #[test]
    fn test_composite_operators() {
        // Test (a + b) * c
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let c = AstNode::Const(3.0f32.into());
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
}
