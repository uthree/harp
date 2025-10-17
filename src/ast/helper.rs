// helper functions for building AST

use crate::ast::{AstNode, AstOp, ConstValue, DType};

impl AstNode {
    pub fn new(op: AstOp) -> Self {
        AstNode {
            op,
            dtype: DType::Unknown,
        }
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
}

// Macro for binary operators
macro_rules! impl_binary_op {
    ($fn_name:ident, $op_variant:ident) => {
        pub fn $fn_name(lhs: AstNode, rhs: AstNode) -> AstNode {
            assert!(
                lhs.dtype == rhs.dtype,
                "{} requires both operands to have the same dtype",
                stringify!($fn_name)
            );
            let dtype = lhs.dtype.clone();
            AstNode::new(AstOp::$op_variant(Box::new(lhs), Box::new(rhs))).with_dtype(dtype)
        }
    };
}

// Macro for unary operators
macro_rules! impl_unary_op {
    ($fn_name:ident, $op_variant:ident) => {
        pub fn $fn_name(operand: AstNode) -> AstNode {
            let dtype = operand.dtype.clone();
            AstNode::new(AstOp::$op_variant(Box::new(operand))).with_dtype(dtype)
        }
    };
}

// Macro for constant constructors
macro_rules! impl_const_fn {
    ($fn_name:ident, $const_variant:ident, $rust_type:ty, $dtype:expr) => {
        pub fn $fn_name(value: $rust_type) -> AstNode {
            AstNode::new(AstOp::Const(ConstValue::$const_variant(value))).with_dtype($dtype)
        }
    };
}

// Binary operators
impl_binary_op!(add, Add);
impl_binary_op!(mul, Mul);
impl_binary_op!(max, Max);
impl_binary_op!(idiv, Idiv);
impl_binary_op!(rem, Rem);

// Unary operators
impl_unary_op!(neg, Neg);
impl_unary_op!(recip, Recip);
impl_unary_op!(sqrt, Sqrt);
impl_unary_op!(sin, Sin);
impl_unary_op!(log2, Log2);
impl_unary_op!(exp2, Exp2);

// Constant constructors
impl_const_fn!(const_isize, Isize, isize, DType::Isize);
impl_const_fn!(const_usize, Usize, usize, DType::Usize);
impl_const_fn!(const_f32, F32, f32, DType::F32);
impl_const_fn!(const_bool, Bool, bool, DType::Bool);

// Macro for From trait implementation
macro_rules! impl_from_primitive {
    ($rust_type:ty, $const_variant:ident, $dtype:expr) => {
        impl From<$rust_type> for AstNode {
            fn from(value: $rust_type) -> Self {
                AstNode::new(AstOp::Const(ConstValue::$const_variant(value))).with_dtype($dtype)
            }
        }
    };
}

// From trait implementations for primitive types
impl_from_primitive!(isize, Isize, DType::Isize);
impl_from_primitive!(usize, Usize, DType::Usize);
impl_from_primitive!(f32, F32, DType::F32);
impl_from_primitive!(bool, Bool, DType::Bool);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_constructors() {
        let isize_node = const_isize(42);
        assert_eq!(isize_node.dtype, DType::Isize);

        let usize_node = const_usize(100);
        assert_eq!(usize_node.dtype, DType::Usize);

        let f32_node = const_f32(3.14);
        assert_eq!(f32_node.dtype, DType::F32);

        let bool_node = const_bool(true);
        assert_eq!(bool_node.dtype, DType::Bool);
    }

    #[test]
    fn test_binary_operators() {
        let lhs = const_f32(1.0);
        let rhs = const_f32(2.0);

        let add_node = add(lhs.clone(), rhs.clone());
        assert_eq!(add_node.dtype, DType::F32);

        let mul_node = mul(lhs.clone(), rhs.clone());
        assert_eq!(mul_node.dtype, DType::F32);

        let max_node = max(lhs.clone(), rhs.clone());
        assert_eq!(max_node.dtype, DType::F32);

        let idiv_node = idiv(lhs.clone(), rhs.clone());
        assert_eq!(idiv_node.dtype, DType::F32);

        let rem_node = rem(lhs, rhs);
        assert_eq!(rem_node.dtype, DType::F32);
    }

    #[test]
    fn test_unary_operators() {
        let operand = const_f32(5.0);

        let neg_node = neg(operand.clone());
        assert_eq!(neg_node.dtype, DType::F32);

        let recip_node = recip(operand.clone());
        assert_eq!(recip_node.dtype, DType::F32);

        let sqrt_node = sqrt(operand.clone());
        assert_eq!(sqrt_node.dtype, DType::F32);

        let sin_node = sin(operand.clone());
        assert_eq!(sin_node.dtype, DType::F32);

        let log2_node = log2(operand.clone());
        assert_eq!(log2_node.dtype, DType::F32);

        let exp2_node = exp2(operand);
        assert_eq!(exp2_node.dtype, DType::F32);
    }

    #[test]
    #[should_panic(expected = "add requires both operands to have the same dtype")]
    fn test_binary_operator_type_mismatch() {
        let lhs = const_f32(1.0);
        let rhs = const_isize(2);
        let _ = add(lhs, rhs); // Should panic because types don't match
    }

    #[test]
    fn test_nested_operations() {
        // Test: (1.0 + 2.0) * 3.0
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let c = const_f32(3.0);

        let sum = add(a, b);
        let product = mul(sum, c);

        assert_eq!(product.dtype, DType::F32);
    }

    #[test]
    fn test_complex_expression() {
        // Test: max(-(1.0 + 2.0), recip(3.0))
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let c = const_f32(3.0);

        let sum = add(a, b);
        let negated = neg(sum);
        let reciprocal = recip(c);
        let result = max(negated, reciprocal);

        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_from_isize() {
        let node: AstNode = 42isize.into();
        assert_eq!(node.dtype, DType::Isize);
    }

    #[test]
    fn test_from_usize() {
        let node: AstNode = 100usize.into();
        assert_eq!(node.dtype, DType::Usize);
    }

    #[test]
    fn test_from_f32() {
        let node: AstNode = 3.14f32.into();
        assert_eq!(node.dtype, DType::F32);
    }

    #[test]
    fn test_from_bool() {
        let node: AstNode = true.into();
        assert_eq!(node.dtype, DType::Bool);
    }

    #[test]
    fn test_implicit_conversion_in_operations() {
        // Test that we can use .into() to convert primitives in operations
        let a: AstNode = 1.0f32.into();
        let b: AstNode = 2.0f32.into();
        let result = add(a, b);
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_into_with_type_inference() {
        // Test using From::from with type inference
        let node = AstNode::from(42isize);
        assert_eq!(node.dtype, DType::Isize);

        let node = AstNode::from(3.14f32);
        assert_eq!(node.dtype, DType::F32);

        let node = AstNode::from(true);
        assert_eq!(node.dtype, DType::Bool);
    }

    #[test]
    fn test_operator_overload_add() {
        // Test addition with operator overloading
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let result = a + b;
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_operator_overload_mul() {
        // Test multiplication with operator overloading
        let a = const_f32(3.0);
        let b = const_f32(4.0);
        let result = a * b;
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_operator_overload_div() {
        // Test division with operator overloading
        let a = const_f32(10.0);
        let b = const_f32(2.0);
        let result = a / b;
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_operator_overload_neg() {
        // Test negation with operator overloading
        let a = const_f32(5.0);
        let result = -a;
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_operator_overload_sub() {
        // Test subtraction with operator overloading (implemented as add + neg)
        let a = const_f32(10.0);
        let b = const_f32(3.0);
        let result = a - b;
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_operator_overload_with_primitives() {
        // Test that operators work with primitives through Into<AstNode>
        let a = const_f32(5.0);
        let result = a + 3.0f32;
        assert_eq!(result.dtype, DType::F32);

        let b = const_isize(10);
        let result = b * 2isize;
        assert_eq!(result.dtype, DType::Isize);
    }

    #[test]
    fn test_operator_overload_complex_expression() {
        // Test: (a + b) * c - d / e
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let c = const_f32(3.0);
        let d = const_f32(8.0);
        let e = const_f32(2.0);

        let result = (a + b) * c - d / e;
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_operator_overload_chained_operations() {
        // Test: a + b + c + d
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let c = const_f32(3.0);
        let d = const_f32(4.0);

        let result = a + b + c + d;
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_operator_overload_negation_chaining() {
        // Test: -(-a)
        let a = const_f32(5.0);
        let result = -(-a);
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_idiv_operator() {
        // Test integer division
        let a = const_isize(10);
        let b = const_isize(3);
        let result = idiv(a, b);
        assert_eq!(result.dtype, DType::Isize);
    }

    #[test]
    fn test_rem_operator() {
        // Test remainder
        let a = const_isize(10);
        let b = const_isize(3);
        let result = rem(a, b);
        assert_eq!(result.dtype, DType::Isize);
    }

    #[test]
    fn test_sqrt_operator() {
        // Test square root
        let a = const_f32(16.0);
        let result = sqrt(a);
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_sin_operator() {
        // Test sine
        let a = const_f32(0.0);
        let result = sin(a);
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_log2_operator() {
        // Test log2
        let a = const_f32(8.0);
        let result = log2(a);
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_exp2_operator() {
        // Test exp2
        let a = const_f32(3.0);
        let result = exp2(a);
        assert_eq!(result.dtype, DType::F32);
    }

    #[test]
    fn test_complex_math_expression() {
        // Test: sqrt(a) + sin(b) * exp2(log2(c))
        let a = const_f32(16.0);
        let b = const_f32(0.5);
        let c = const_f32(4.0);

        let sqrt_a = sqrt(a);
        let sin_b = sin(b);
        let log2_c = log2(c);
        let exp2_log2_c = exp2(log2_c);
        let product = mul(sin_b, exp2_log2_c);
        let result = add(sqrt_a, product);

        assert_eq!(result.dtype, DType::F32);
    }
}
