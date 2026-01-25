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

#[test]
fn test_scan_lowering() {
    // Test cumulative sum lowering
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
    let cumsum_result = x.cumsum(1).with_name("cumsum_result");

    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&[cumsum_result]);

    assert!(matches!(program, crate::ast::AstNode::Program { .. }));
}

#[test]
fn test_scan_shape_preservation() {
    // Verify that scan preserves shape (unlike reduce)
    let x = input(vec![Expr::Const(8), Expr::Const(16)], DType::F32);

    // cumsum should preserve shape
    let y = x.cumsum(0);
    assert_eq!(y.shape(), vec![Expr::Const(8), Expr::Const(16)]);

    // cumprod should preserve shape
    let z = x.cumprod(1);
    assert_eq!(z.shape(), vec![Expr::Const(8), Expr::Const(16)]);

    // cummax should preserve shape
    let w = x.cummax(0);
    assert_eq!(w.shape(), vec![Expr::Const(8), Expr::Const(16)]);
}

#[test]
fn test_scan_loop_generation() {
    use crate::ast::Literal;

    let loop_gen = LoopGenerator::new();
    let shape = vec![Expr::Const(4), Expr::Const(8)];

    let output_ptr = crate::ast::AstNode::Var("output".to_string());
    let output_idx = crate::ast::AstNode::Var("idx".to_string());
    let identity = crate::ast::AstNode::Const(Literal::F32(0.0));
    let combine = crate::ast::AstNode::Add(
        Box::new(crate::ast::AstNode::Var("acc".to_string())),
        Box::new(crate::ast::AstNode::Var("val".to_string())),
    );

    let scan_loops = loop_gen.generate_scan(
        &shape,
        1, // scan along axis 1
        output_ptr,
        output_idx,
        "acc",
        &DType::F32,
        identity,
        combine,
    );

    // Should be nested Range structures
    assert!(matches!(scan_loops, crate::ast::AstNode::Range { .. }));
}
