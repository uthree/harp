use harp::{ast::*, backend::KernelDetails, opt::ast::*};

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
    let x = AstNode::var("x").with_type(DType::F32);
    let y = AstNode::var("y").with_type(DType::F32);
    let z = AstNode::var("z").with_type(DType::F32);

    // (x * 1.0 + 0.0) * (y * 0.0 + z)
    let target = (x.clone() * 1.0f32 + 0.0f32) * (y * 0.0f32 + z.clone());
    let expected = x * z;

    let optimizer = AlgebraicSimplification::new();
    let dummy_details = KernelDetails::default();
    let result = optimizer.optimize(target, &dummy_details);

    assert_eq!(result, expected);
}

#[test]
fn test_algebraic_simplification_integer() {
    setup_logger();
    let optimizer = AlgebraicSimplification::new();
    let dummy_details = KernelDetails::default();

    // --- Test various integer types ---
    let types_to_test = vec![
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U64,
        DType::ISize,
        DType::USize,
    ];

    for dtype in types_to_test {
        let x = AstNode::var("x").with_type(dtype.clone());

        let one: AstNode = match dtype {
            DType::I32 => 1i32.into(),
            DType::I64 => 1i64.into(),
            DType::U32 => 1u32.into(),
            DType::U64 => 1u64.into(),
            DType::ISize => 1isize.into(),
            DType::USize => 1usize.into(),
            _ => panic!("Unsupported integer type for test"),
        };

        let zero: AstNode = match dtype {
            DType::I32 => 0i32.into(),
            DType::I64 => 0i64.into(),
            DType::U32 => 0u32.into(),
            DType::U64 => 0u64.into(),
            DType::ISize => 0isize.into(),
            DType::USize => 0usize.into(),
            _ => panic!("Unsupported integer type for test"),
        };

        // Test: x * 1 = x
        let target_mul_one = x.clone() * one.clone();
        let result_mul_one = optimizer.optimize(target_mul_one, &dummy_details);
        assert_eq!(result_mul_one, x, "Failed for type {:?}", dtype);

        // Test: x + 0 = x
        let target_add_zero = x.clone() + zero.clone();
        let result_add_zero = optimizer.optimize(target_add_zero, &dummy_details);
        assert_eq!(result_add_zero, x, "Failed for type {:?}", dtype);

        // Test: x * 0 = 0
        let target_mul_zero = x.clone() * zero.clone();
        let result_mul_zero = optimizer.optimize(target_mul_zero, &dummy_details);
        assert_eq!(result_mul_zero, zero, "Failed for type {:?}", dtype);
    }
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

#[test]
fn test_constant_folding() {
    setup_logger();
    let optimizer = AlgebraicSimplification::new();
    let dummy_details = KernelDetails::default();

    // 2 + 3 = 5
    let target_add = AstNode::from(2i32) + AstNode::from(3i32);
    let expected_add = AstNode::from(5i32);
    let result_add = optimizer.optimize(target_add, &dummy_details);
    assert_eq!(result_add, expected_add);

    // 2.0 * 3.0 = 6.0
    let target_mul = AstNode::from(2.0f32) * AstNode::from(3.0f32);
    let expected_mul = AstNode::from(6.0f32);
    let result_mul = optimizer.optimize(target_mul, &dummy_details);
    assert_eq!(result_mul, expected_mul);
}

#[test]
fn test_associative_and_distributive_suggester() {
    setup_logger();
    let suggester = AssociativeAndDistributiveLaws;
    let a = AstNode::var("a");
    let b = AstNode::var("b");
    let c = AstNode::var("c");

    // Test distributive expand: a * (b + c)
    let target_dist_expand = a.clone() * (b.clone() + c.clone());
    let suggestions = suggester.suggest_optimizations(&target_dist_expand);
    assert_eq!(suggestions.len(), 1);
    let expected_expand = (a.clone() * b.clone()) + (a.clone() * c.clone());
    assert_eq!(suggestions[0], expected_expand);

    // Test associative add: (a + b) + c
    let target_assoc_add = (a.clone() + b.clone()) + c.clone();
    let suggestions = suggester.suggest_optimizations(&target_assoc_add);
    assert_eq!(suggestions.len(), 1);
    let expected_assoc = a.clone() + (b.clone() + c.clone());
    assert_eq!(suggestions[0], expected_assoc);
}
