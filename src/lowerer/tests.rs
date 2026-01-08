//! Integration tests for the lowerer module

use super::*;
use crate::ast::DType;
use crate::graph::{Expr, input};

#[test]
fn test_full_pipeline() {
    // Create computation graph
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("input_x");
    let y = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("input_y");

    // Build computation: z = (x + y) * x
    let z = (&(&x + &y) * &x).with_name("output");

    // Lower to AST
    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[z]);

    // Verify program structure
    match program {
        crate::ast::AstNode::Program { functions, .. } => {
            // Should have at least one kernel
            assert!(!functions.is_empty());
        }
        _ => panic!("Expected Program node"),
    }
}

#[test]
fn test_view_lowering() {
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
    let transposed = x.permute(&[1, 0]).with_name("transposed");

    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[transposed]);

    assert!(matches!(program, crate::ast::AstNode::Program { .. }));
}

#[test]
fn test_reduction_lowering() {
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

    // Sum along axis 1
    let sum_result = x.sum(1).with_name("sum_result");

    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[sum_result]);

    assert!(matches!(program, crate::ast::AstNode::Program { .. }));
}

#[test]
fn test_fusion_applied() {
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

    // This should potentially fuse: elementwise followed by reduce
    let doubled = &x * &x;
    let summed = doubled.sum(1);

    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[summed]);

    assert!(matches!(program, crate::ast::AstNode::Program { .. }));
}

#[test]
fn test_index_generation() {
    let idx_gen = IndexGenerator::new();

    // Test contiguous view
    let view = crate::graph::View::contiguous(vec![Expr::Const(32), Expr::Const(64)]);
    let idx = idx_gen.view_to_index(&view);

    // Should generate index expression
    assert!(matches!(idx, crate::ast::AstNode::Add(_, _)));
}

#[test]
fn test_loop_generation() {
    use crate::ast::Literal;

    let loop_gen = LoopGenerator::new();
    let shape = vec![Expr::Const(32), Expr::Const(64)];

    let body = crate::ast::AstNode::Const(Literal::I64(0));
    let loops = loop_gen.generate_loops(&shape, body);

    // Should be nested Range structures
    assert!(matches!(loops, crate::ast::AstNode::Range { .. }));
}
