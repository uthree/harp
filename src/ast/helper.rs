use super::{AstNode, DType, Literal};
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

// Helper functions for constructing AST nodes
impl AstNode {
    /// Create a constant node from a literal value
    pub fn const_node(literal: Literal) -> Self {
        AstNode::Const(literal)
    }

    /// Create a constant f32 node
    pub fn const_f32(value: f32) -> Self {
        AstNode::Const(Literal::F32(value))
    }

    /// Create a constant isize node
    pub fn const_isize(value: isize) -> Self {
        AstNode::Const(Literal::Isize(value))
    }

    /// Create a constant usize node
    pub fn const_usize(value: usize) -> Self {
        AstNode::Const(Literal::Usize(value))
    }

    /// Create a max node: max(a, b)
    pub fn max(a: AstNode, b: AstNode) -> Self {
        AstNode::Max(Box::new(a), Box::new(b))
    }

    /// Create an integer division node: a / b
    pub fn idiv(a: AstNode, b: AstNode) -> Self {
        AstNode::Idiv(Box::new(a), Box::new(b))
    }

    /// Create a reciprocal node: 1 / a
    pub fn recip(a: AstNode) -> Self {
        AstNode::Recip(Box::new(a))
    }

    /// Create a square root node: sqrt(a)
    pub fn sqrt(a: AstNode) -> Self {
        AstNode::Sqrt(Box::new(a))
    }

    /// Create a log2 node: log2(a)
    pub fn log2(a: AstNode) -> Self {
        AstNode::Log2(Box::new(a))
    }

    /// Create an exp2 node: 2^a
    pub fn exp2(a: AstNode) -> Self {
        AstNode::Exp2(Box::new(a))
    }

    /// Create a sine node: sin(a)
    pub fn sin(a: AstNode) -> Self {
        AstNode::Sin(Box::new(a))
    }

    /// Create a cast node: cast a to dtype
    pub fn cast(a: AstNode, dtype: DType) -> Self {
        AstNode::Cast(Box::new(a), dtype)
    }
}

// Convenience free functions for more concise AST construction

/// Create a max node: max(a, b)
pub fn max(a: AstNode, b: AstNode) -> AstNode {
    AstNode::max(a, b)
}

/// Create an integer division node: a / b
pub fn idiv(a: AstNode, b: AstNode) -> AstNode {
    AstNode::idiv(a, b)
}

/// Create a reciprocal node: 1 / a
pub fn recip(a: AstNode) -> AstNode {
    AstNode::recip(a)
}

/// Create a square root node: sqrt(a)
pub fn sqrt(a: AstNode) -> AstNode {
    AstNode::sqrt(a)
}

/// Create a log2 node: log2(a)
pub fn log2(a: AstNode) -> AstNode {
    AstNode::log2(a)
}

/// Create an exp2 node: 2^a
pub fn exp2(a: AstNode) -> AstNode {
    AstNode::exp2(a)
}

/// Create a sine node: sin(a)
pub fn sin(a: AstNode) -> AstNode {
    AstNode::sin(a)
}

/// Create a cast node: cast a to dtype
pub fn cast(a: AstNode, dtype: DType) -> AstNode {
    AstNode::cast(a, dtype)
}

/// Create a constant f32 node
pub fn const_f32(value: f32) -> AstNode {
    AstNode::const_f32(value)
}

/// Create a constant isize node
pub fn const_isize(value: isize) -> AstNode {
    AstNode::const_isize(value)
}

/// Create a constant usize node
pub fn const_usize(value: usize) -> AstNode {
    AstNode::const_usize(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_helpers() {
        // Test constant creation helpers
        let f32_node = const_f32(3.14);
        match f32_node {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 3.14),
            _ => panic!("Expected F32 constant"),
        }

        let isize_node = const_isize(42);
        match isize_node {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Isize constant"),
        }

        let usize_node = const_usize(100);
        match usize_node {
            AstNode::Const(Literal::Usize(v)) => assert_eq!(v, 100),
            _ => panic!("Expected Usize constant"),
        }
    }

    #[test]
    fn test_binary_ops() {
        // Test binary operation using operator overloading
        let a = const_f32(1.0);
        let b = const_f32(2.0);

        let add_node = a.clone() + b.clone();
        match add_node {
            AstNode::Add(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::F32(l)), AstNode::Const(Literal::F32(r))) => {
                    assert_eq!(l, 1.0);
                    assert_eq!(r, 2.0);
                }
                _ => panic!("Expected F32 constants in Add node"),
            },
            _ => panic!("Expected Add node"),
        }

        let mul_node = a.clone() * b.clone();
        match mul_node {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }

        let max_node = max(a.clone(), b.clone());
        match max_node {
            AstNode::Max(_, _) => {}
            _ => panic!("Expected Max node"),
        }

        let rem_node = a.clone() % b.clone();
        match rem_node {
            AstNode::Rem(_, _) => {}
            _ => panic!("Expected Rem node"),
        }

        let idiv_node = idiv(a.clone(), b.clone());
        match idiv_node {
            AstNode::Idiv(_, _) => {}
            _ => panic!("Expected Idiv node"),
        }
    }

    #[test]
    fn test_unary_ops() {
        // Test unary operation helpers
        let a = const_f32(4.0);

        let recip_node = recip(a.clone());
        match recip_node {
            AstNode::Recip(_) => {}
            _ => panic!("Expected Recip node"),
        }

        let sqrt_node = sqrt(a.clone());
        match sqrt_node {
            AstNode::Sqrt(_) => {}
            _ => panic!("Expected Sqrt node"),
        }

        let log2_node = log2(a.clone());
        match log2_node {
            AstNode::Log2(_) => {}
            _ => panic!("Expected Log2 node"),
        }

        let exp2_node = exp2(a.clone());
        match exp2_node {
            AstNode::Exp2(_) => {}
            _ => panic!("Expected Exp2 node"),
        }

        let sin_node = sin(a.clone());
        match sin_node {
            AstNode::Sin(_) => {}
            _ => panic!("Expected Sin node"),
        }
    }

    #[test]
    fn test_cast() {
        let a = const_f32(3.14);
        let cast_node = cast(a, DType::Isize);
        match cast_node {
            AstNode::Cast(_, dtype) => match dtype {
                DType::Isize => {}
                _ => panic!("Expected Isize dtype"),
            },
            _ => panic!("Expected Cast node"),
        }
    }

    #[test]
    fn test_composite_expression() {
        // Test building a composite expression: (a + b) * c using operator overloading
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let c = const_f32(3.0);

        let product = (a + b) * c;

        match product {
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
    fn test_operator_overloading() {
        // Test that operator overloading works
        let a = AstNode::const_f32(1.0);
        let b = AstNode::const_f32(2.0);
        let sum = a + b;

        match sum {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node"),
        }

        let a = AstNode::const_f32(3.0);
        let b = AstNode::const_f32(4.0);
        let product = a * b;

        match product {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }

        let a = AstNode::const_f32(5.0);
        let b = AstNode::const_f32(2.0);
        let remainder = a % b;

        match remainder {
            AstNode::Rem(_, _) => {}
            _ => panic!("Expected Rem node"),
        }
    }
}
