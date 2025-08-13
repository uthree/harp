// tests/c_backend_render.rs

use harp::ast::{AstNode, DType};
use harp::backend::Renderer;
use harp::backend::c::CRenderer;
use std::f32;

fn setup_logger() {
    // Initialize the logger for tests, ignoring errors if it's already set up
    let _ = env_logger::builder().is_test(true).try_init();
}

// Helper function to render an AST node and compare it with the expected output.
fn assert_render(node: AstNode, expected: &str) {
    setup_logger();
    let mut renderer = CRenderer::new();
    let rendered_code = renderer.render(node);
    // Normalize whitespace and remove initial headers for easier comparison
    let cleaned_code = rendered_code
        .lines()
        .skip(5) // Skip header lines
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let cleaned_expected = expected
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(cleaned_code, cleaned_expected);
}

/// Tests that a simple function definition is rendered correctly.
#[test]
fn test_render_simple_function() {
    let ast = AstNode::func_def(
        "test_func",
        vec![
            ("a".to_string(), DType::Ptr(Box::new(DType::F32))),
            ("b".to_string(), DType::Ptr(Box::new(DType::F32))),
        ],
        vec![],
    );
    let expected = r#"
void test_func(float* a, float* b) { }"#;
    assert_render(ast, expected);
}

/// Tests that an addition operation is rendered correctly.
#[test]
fn test_render_add() {
    let ast = AstNode::var("a").with_type(DType::F32) + AstNode::var("b").with_type(DType::F32);
    let expected = "(a + b)";
    assert_render(ast, expected);
}

/// Tests that a multiplication operation is rendered correctly.
#[test]
fn test_render_mul() {
    let ast = AstNode::var("a").with_type(DType::F32) * AstNode::var("b").with_type(DType::F32);
    let expected = "(a * b)";
    assert_render(ast, expected);
}

/// Tests that a max operation is rendered correctly.
#[test]
fn test_render_max() {
    let ast = AstNode::var("a")
        .with_type(DType::F32)
        .max(AstNode::var("b").with_type(DType::F32));
    let expected = "fmax(a, b)";
    assert_render(ast, expected);
}

/// Tests that a constant is rendered correctly.
#[test]
fn test_render_const() {
    assert_render(f32::consts::PI.into(), "3.1415927");
    assert_render((42i8).into(), "42");
    assert_render((123u32).into(), "123");
    assert_render((9999999999i64).into(), "9999999999");
}

/// Tests that an assignment operation is rendered correctly.
#[test]
fn test_render_assign() {
    let ast = AstNode::assign(AstNode::var("x").with_type(DType::I32), 42i32.into());
    let expected = "x = 42;";
    assert_render(ast, expected);
}

/// Tests that a for loop is rendered correctly.
#[test]
fn test_render_for_loop() {
    let ast = AstNode::range(
        "i".to_string(),
        AstNode::var("N").with_type(DType::I32),
        vec![],
        false,
    );
    let expected = r#"
for (size_t i = 0; i < N; i++) { }"#;
    assert_render(ast, expected);
}

/// Tests that a buffer index operation is rendered correctly.
#[test]
fn test_render_buffer_index() {
    let ast = AstNode::var("data")
        .with_type(DType::Ptr(Box::new(DType::F32)))
        .buffer_index(AstNode::var("i").with_type(DType::I32));
    let expected = "(data)[i]";
    assert_render(ast, expected);
}

/// Tests that a store operation is rendered correctly.
#[test]
fn test_render_store() {
    let ast = AstNode::store(
        AstNode::var("output")
            .with_type(DType::Ptr(Box::new(DType::F32)))
            .buffer_index(AstNode::var("i").with_type(DType::I32)),
        AstNode::var("value").with_type(DType::F32),
    );
    let expected = "(output)[i] = value;";
    assert_render(ast, expected);
}
