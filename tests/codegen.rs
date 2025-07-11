use harp::backend::codegen::CodeGenerator;
use harp::backend::c::CRenderer;
use harp::node::{constant, variable};

#[test]
fn test_simple_codegen() {
    let a = variable("a");
    let b = constant(2.0f32);
    let graph = a + b;

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    let expected = r#"float compute() {
    float v0 = (a + 2.0);
    return v0;
}"#;
    assert_eq!(result, expected);
}

#[test]
fn test_fused_op_codegen() {
    let a = variable("a");
    let b = variable("b");
    let graph = a - b; // OpSub is a FusedOp

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    // The fallback graph is (a + (b * -1.0))
    let expected = r#"float compute() {
    float v0 = (b * -1.0);
    float v1 = (a + v0);
    return v1;
}"#;
    assert_eq!(result, expected);
}

#[test]
fn test_complex_graph_codegen() {
    let a = variable("a");
    let b = constant(2.0f32);
    let c = constant(3.0f32);
    let graph = (a + b) * c;

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    let expected = r#"float compute() {
    float v0 = (a + 2.0);
    float v1 = (v0 * 3.0);
    return v1;
}"#;
    assert_eq!(result, expected);
}
