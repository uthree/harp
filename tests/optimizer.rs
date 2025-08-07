use harp::{
    ast::{AstNode, DType},
    opt::algebraic_simplification,
};

fn setup_logger() {
    // Initialize the logger for tests, ignoring errors if it's already set up
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_algebraic_simplification_simple() {
    setup_logger();
    let optimizer = algebraic_simplification();
    let x = AstNode::var("x").with_type(DType::F32);
    let y = AstNode::var("y").with_type(DType::F64); // Different type for testing

    // Test cases
    let cases = vec![
        // x + 0 -> x
        (x.clone() + AstNode::from(0.0f32), x.clone()),
        // 0 + y -> y
        (AstNode::from(0.0f64) + y.clone(), y.clone()),
        // x - 0 -> x
        (x.clone() - AstNode::from(0.0f32), x.clone()),
        // y * 1 -> y
        (y.clone() * AstNode::from(1.0f64), y.clone()),
        // 1 * x -> x
        (AstNode::from(1.0f32) * x.clone(), x.clone()),
        // x * 0 -> 0
        (
            x.clone() * AstNode::from(0.0f32),
            AstNode::from(0.0f32).cast(x.clone().dtype),
        ),
        // 0 * y -> 0
        (
            AstNode::from(0.0f64) * y.clone(),
            AstNode::from(0.0f32).cast(y.clone().dtype),
        ),
        // x / 1 -> x
        (x.clone() / AstNode::from(1.0f32), x.clone()),
    ];

    for (input, expected) in cases {
        let result = optimizer.apply(input.clone());
        assert_eq!(
            result, expected,
            "\nFailed on input: {:?}\nExpected: {:?}\nGot:      {:?}",
            input, expected, result
        );
    }
}

#[test]
fn test_algebraic_simplification_complex() {
    setup_logger();
    let optimizer = algebraic_simplification();
    let x = AstNode::var("x").with_type(DType::F32);

    // (x + 0) * 1 -> x
    let input = (x.clone() + AstNode::from(0.0f32)) * AstNode::from(1.0f32);
    let expected = x.clone();
    let result = optimizer.apply(input.clone());
    assert_eq!(
        result, expected,
        "\nFailed on input: {:?}\nExpected: {:?}\nGot:      {:?}",
        input, expected, result
    );

    // (y * 1) - 0 -> y
    let y = AstNode::var("y").with_type(DType::F64);
    let input = (y.clone() * AstNode::from(1.0f64)) - AstNode::from(0.0f64);
    let expected = y.clone();
    let result = optimizer.apply(input.clone());
    assert_eq!(
        result, expected,
        "\nFailed on input: {:?}\nExpected: {:?}\nGot:      {:?}",
        input, expected, result
    );
}
