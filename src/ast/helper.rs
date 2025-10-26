use super::{AstNode, DType};

// Convenience free functions for AST construction

/// Macro to generate binary operation helper functions
macro_rules! impl_binary_helper {
    ($fn_name:ident, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name(a: AstNode, b: AstNode) -> AstNode {
            AstNode::$variant(Box::new(a), Box::new(b))
        }
    };
}

/// Macro to generate unary operation helper functions
macro_rules! impl_unary_helper {
    ($fn_name:ident, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name(a: AstNode) -> AstNode {
            AstNode::$variant(Box::new(a))
        }
    };
}

// Binary operation helpers
impl_binary_helper!(max, Max, "Create a max node: max(a, b)");
impl_binary_helper!(idiv, Idiv, "Create an integer division node: a / b");

// Unary operation helpers
impl_unary_helper!(recip, Recip, "Create a reciprocal node: 1 / a");
impl_unary_helper!(sqrt, Sqrt, "Create a square root node: sqrt(a)");
impl_unary_helper!(log2, Log2, "Create a log2 node: log2(a)");
impl_unary_helper!(exp2, Exp2, "Create an exp2 node: 2^a");
impl_unary_helper!(sin, Sin, "Create a sine node: sin(a)");

/// Create a cast node: cast a to dtype
pub fn cast(a: AstNode, dtype: DType) -> AstNode {
    AstNode::Cast(Box::new(a), dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_const_creation() {
        // Test constant creation using Into
        let f32_node = AstNode::Const(3.14f32.into());
        match f32_node {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 3.14),
            _ => panic!("Expected F32 constant"),
        }

        let isize_node = AstNode::Const(42isize.into());
        match isize_node {
            AstNode::Const(Literal::Isize(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Isize constant"),
        }

        let usize_node = AstNode::Const(100usize.into());
        match usize_node {
            AstNode::Const(Literal::Usize(v)) => assert_eq!(v, 100),
            _ => panic!("Expected Usize constant"),
        }
    }

    #[test]
    fn test_binary_ops() {
        // Test binary operation using operator overloading
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());

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
        let a = AstNode::Const(4.0f32.into());

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
        let a = AstNode::Const(3.14f32.into());
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
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let c = AstNode::Const(3.0f32.into());

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
}
