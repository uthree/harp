use harp::{ast::*, backend::c::CBackend, backend::KernelDetails, opt::ast::*};

fn setup_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_algebraic_simplification_simple() {
    setup_logger();
    let x = AstNode::var("x").with_type(DType::F32);
    let target = (x.clone() * 1.0f32) + 0.0f32;
    let expected = x;

    let optimizer = AlgebraicSimplification::new();
    let dummy_details = KernelDetails::default();
    let result = optimizer.optimize(target, &dummy_details);

    assert_eq!(result, expected);
}

#[test]
fn test_algebraic_simplification_complex() {
    setup_logger();
    let x = AstNode::var("x").with_type(DType::F64);
    let y = AstNode::var("y").with_type(DType::F64);
    let z = AstNode::var("z").with_type(DType::F64);

    // (x * 1.0 + 0.0) * (y * 0.0 + z)
    let target = (x.clone() * 1.0f64 + 0.0f64) * (y * 0.0f64 + z.clone());
    let expected = x * z;

    let optimizer = AlgebraicSimplification::new();
    let dummy_details = KernelDetails::default();
    let result = optimizer.optimize(target, &dummy_details);

    assert_eq!(result, expected);
}

#[test]
fn test_cast_simplification_redundant() {
    setup_logger();
    let x = AstNode::var("x").with_type(DType::F32);
    // (float)((float)x)
    let target = AstNode::new(
        AstOp::Cast(DType::F32),
        vec![AstNode::new(
            AstOp::Cast(DType::F32),
            vec![x.clone()],
            DType::F32,
        )],
        DType::F32,
    );
    let expected = x;

    let optimizer = CastSimplification;
    let dummy_details = KernelDetails::default();
    let result = optimizer.optimize(target, &dummy_details);

    assert_eq!(result, expected);
}

#[test]
fn test_cast_simplification_constant_folding() {
    setup_logger();
    // (int)3.14f32
    let target = AstNode::new(
        AstOp::Cast(DType::I32),
        vec![AstNode::from(3.14f32)],
        DType::I32,
    );
    let expected = AstNode::from(3i32);

    let optimizer = CastSimplification;
    let dummy_details = KernelDetails::default();
    let result = optimizer.optimize(target, &dummy_details);

    assert_eq!(result, expected);
}