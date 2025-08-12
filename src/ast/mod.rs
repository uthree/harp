//! This module defines the Abstract Syntax Tree (AST) for the computation.
//!
//! The AST is a lower-level, more explicit representation of the computation
//! graph. It uses `AstNode` as its fundamental building block to represent
//! operations, variables, and control flow structures like loops and blocks.

pub mod builder;
pub mod dtype;
pub mod node;
pub mod op;

pub use dtype::{Const, DType};
pub use node::AstNode;
pub use op::AstOp;

#[cfg(test)]
mod tests {

    use crate::ast::{AstNode, AstOp, DType};
    #[test]
    fn test_unary_ops() {
        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);

        let neg_a = -a.clone();
        assert_eq!(neg_a.op, AstOp::Neg);
        assert_eq!(neg_a.src.len(), 1);
        assert_eq!(neg_a.src[0], a);

        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let sqrt_a = a.clone().sqrt();
        assert_eq!(sqrt_a.op, AstOp::Sqrt);
        assert_eq!(sqrt_a.src.len(), 1);
        assert_eq!(sqrt_a.src[0], a);

        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let sin_a = a.clone().sin();
        assert_eq!(sin_a.op, AstOp::Sin);
        assert_eq!(sin_a.src.len(), 1);
        assert_eq!(sin_a.src[0], a);
    }

    #[test]
    fn test_binary_ops() {
        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(AstOp::Var("b".to_string()), vec![], DType::Any);

        let add_ab = a.clone() + b.clone();
        assert_eq!(add_ab.op, AstOp::Add);
        assert_eq!(add_ab.src.len(), 2);
        assert_eq!(add_ab.src[0], a);
        assert_eq!(add_ab.src[1], b);

        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(AstOp::Var("b".to_string()), vec![], DType::Any);
        let sub_ab = a.clone() - b.clone();
        assert_eq!(sub_ab.op, AstOp::Add); // sub is implemented as a + (-b)
        assert_eq!(sub_ab.src.len(), 2);
        assert_eq!(sub_ab.src[0], a);
        assert_eq!(sub_ab.src[1].op, AstOp::Neg);
        assert_eq!(sub_ab.src[1].src[0], b);

        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(AstOp::Var("b".to_string()), vec![], DType::Any);
        let mul_ab = a.clone() * b.clone();
        assert_eq!(mul_ab.op, AstOp::Mul);
        assert_eq!(mul_ab.src.len(), 2);
        assert_eq!(mul_ab.src[0], a);
        assert_eq!(mul_ab.src[1], b);

        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(AstOp::Var("b".to_string()), vec![], DType::Any);
        let div_ab = a.clone() / b.clone();
        assert_eq!(div_ab.op, AstOp::Mul); // div is implemented as a * (1/b)
        assert_eq!(div_ab.src.len(), 2);
        assert_eq!(div_ab.src[0], a);
        assert_eq!(div_ab.src[1].op, AstOp::Recip);
        assert_eq!(div_ab.src[1].src[0], b);

        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(AstOp::Var("b".to_string()), vec![], DType::Any);
        let rem_ab = a.clone() % b.clone();
        assert_eq!(rem_ab.op, AstOp::Rem);
        assert_eq!(rem_ab.src.len(), 2);
        assert_eq!(rem_ab.src[0], a);
        assert_eq!(rem_ab.src[1], b);

        let a = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let b = AstNode::new(AstOp::Var("b".to_string()), vec![], DType::Any);
        let max_ab = a.clone().max(b.clone());
        assert_eq!(max_ab.op, AstOp::Max);
        assert_eq!(max_ab.src.len(), 2);
        assert_eq!(max_ab.src[0], a);
        assert_eq!(max_ab.src[1], b);
    }

    #[test]
    fn test_complex_expression() {
        let a = AstNode::var("a").with_type(DType::F64);
        let b = AstNode::var("b").with_type(DType::F64);
        let c: AstNode = 2.0f64.into();

        // (a + b) * c
        let expr = (a.clone() + b.clone()) * c.clone();

        assert_eq!(expr.op, AstOp::Mul);
        assert_eq!(expr.src.len(), 2);
        assert_eq!(expr.src[1], c);

        let add_expr = &expr.src[0];
        assert_eq!(add_expr.op, AstOp::Add);
        assert_eq!(add_expr.src.len(), 2);
        assert_eq!(add_expr.src[0], a);
        assert_eq!(add_expr.src[1], b);
    }

    #[test]
    fn test_partial_eq_ignores_id() {
        let node1 = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);
        let node2 = AstNode::new(AstOp::Var("a".to_string()), vec![], DType::Any);

        // IDs should be different
        assert_ne!(node1.id, node2.id);
        // But the nodes should be considered equal
        assert_eq!(node1, node2);

        let node3 = AstNode::new(AstOp::Var("b".to_string()), vec![], DType::Any);
        assert_ne!(node1, node3);
    }

    #[test]
    fn test_dtype_matches() {
        use super::DType;

        // Exact matches
        assert!(DType::F32.matches(&DType::F32));
        assert!(!DType::F32.matches(&DType::F64));

        // Any
        assert!(DType::Any.matches(&DType::F32));
        assert!(DType::Any.matches(&DType::I64));
        assert!(DType::Any.matches(&DType::U8));
        assert!(DType::Any.matches(&DType::Ptr(Box::new(DType::F32))));

        // Real
        assert!(DType::Real.matches(&DType::F32));
        assert!(DType::Real.matches(&DType::F64));
        assert!(!DType::Real.matches(&DType::I32));

        // Natural
        assert!(DType::Natural.matches(&DType::U8));
        assert!(DType::Natural.matches(&DType::U64));
        assert!(!DType::Natural.matches(&DType::I8));
        assert!(!DType::Natural.matches(&DType::F32));

        // Integer
        assert!(DType::Integer.matches(&DType::I32));
        assert!(DType::Integer.matches(&DType::U16));
        assert!(!DType::Integer.matches(&DType::F64));

        // Pointer
        let p_f32 = DType::Ptr(Box::new(DType::F32));
        let p_f64 = DType::Ptr(Box::new(DType::F64));
        let p_any = DType::Ptr(Box::new(DType::Any));
        assert!(p_f32.matches(&p_f32));
        assert!(!p_f32.matches(&p_f64));
        assert!(p_any.matches(&p_f32));
        assert!(!p_f32.matches(&p_any)); // A specific type does not match a general one

        // USize
        assert!(DType::Natural.matches(&DType::USize));
        assert!(DType::Integer.matches(&DType::USize));
        assert!(!DType::Real.matches(&DType::USize));
    }

    #[test]
    fn test_op_type_check_ok() {
        let f1 = AstNode::var("f1").with_type(DType::F32);
        let f2 = AstNode::var("f2").with_type(DType::F64);
        let i1 = AstNode::var("i1").with_type(DType::I32);

        // Real + Real -> Real
        let add_ff = f1.clone() + f2.clone();
        assert!(add_ff.dtype.is_real());

        // Real + Integer -> Real
        let add_fi = f1.clone() + i1.clone();
        assert!(add_fi.dtype.is_real());

        // Neg on Real
        let neg_f = -f1;
        assert!(neg_f.dtype.is_real());
    }

    #[test]
    #[should_panic]
    fn test_op_type_check_panic_add() {
        let p1 = AstNode::var("p1").with_type(DType::Ptr(Box::new(DType::F32)));
        let i1 = AstNode::var("i1").with_type(DType::I32);
        // Ptr + Integer should panic
        let _ = p1 + i1;
    }

    #[test]
    #[should_panic]
    fn test_op_type_check_panic_neg() {
        let p1 = AstNode::var("p1").with_type(DType::Ptr(Box::new(DType::F32)));
        // Neg on Ptr should panic
        let _ = -p1;
    }

    #[test]
    fn test_implicit_cast() {
        let i1 = AstNode::var("i1").with_type(DType::I32);
        let f1 = AstNode::var("f1").with_type(DType::F32);

        // i1 + f1 should result in Cast(I32 as F32) + F32
        let result = i1.clone() + f1.clone();

        assert_eq!(result.op, AstOp::Add);
        assert_eq!(result.dtype, DType::F32);

        let lhs = &result.src[0];
        let rhs = &result.src[1];

        // Check that the integer was cast to float
        assert_eq!(lhs.op, AstOp::Cast(DType::F32));
        assert_eq!(lhs.dtype, DType::F32);
        assert_eq!(lhs.src[0], i1); // Original i1 inside the cast

        // Check that the float remains unchanged
        assert_eq!(rhs, &f1);
    }

    #[test]
    fn test_ast_add_assign() {
        let mut a = AstNode::var("a").with_type(DType::F32);
        let b = AstNode::var("b").with_type(DType::F32);
        let c = a.clone() + b.clone();
        a += b;
        assert_eq!(a, c);
    }

    #[test]
    fn test_ast_sub_assign() {
        let mut a = AstNode::var("a").with_type(DType::F32);
        let b = AstNode::var("b").with_type(DType::F32);
        let c = a.clone() - b.clone();
        a -= b;
        assert_eq!(a, c);
    }

    #[test]
    fn test_func_def_and_call() {
        let a = AstNode::var("a");
        let b = AstNode::var("b");
        let body = vec![a.clone() + b.clone()];
        let args = vec![("a".to_string(), DType::F32), ("b".to_string(), DType::F32)];
        let func = AstNode::func_def("my_add", args.clone(), body);

        if let AstOp::Func {
            name,
            args: func_args,
        } = &func.op
        {
            assert_eq!(name, "my_add");
            assert_eq!(*func_args, args);
        } else {
            panic!("Expected a function definition");
        }

        let x = AstNode::var("x");
        let y = 1.0f32.into();
        let call = AstNode::call("my_add", vec![x, y]);

        if let AstOp::Call(name) = &call.op {
            assert_eq!(name, "my_add");
            assert_eq!(call.src.len(), 2);
        } else {
            panic!("Expected a function call");
        }
    }
}
