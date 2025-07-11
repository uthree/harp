use harp::backend::codegen::CodeGenerator;
use harp::backend::c::CRenderer;
use harp::node::{self, constant, variable, Node};
use harp::op::{Load, Loop, LoopVariable, Store};

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

#[test]
fn test_loop_codegen() {
    let loop_var = Node::new(LoopVariable, vec![]);
    let loop_body = loop_var + constant(1.0f32);
    let count = constant(10);
    let graph = Node::new(
        Loop {
            count: count.clone(),
            body: loop_body,
        },
        vec![count],
    );

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    let expected = r#"float compute() {
    for (int i = 0; i < 10; ++i) {
        float v0 = (i + 1.0);
    }
    return loop_result;
}"#;
    assert_eq!(result, expected);
}

#[test]
fn test_load_codegen() {
    let index = constant(0);
    let load_op = Load("input".to_string(), 10);
    let graph = Node::new(load_op, vec![index]) + constant(1.0f32);

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    let expected = r#"float compute() {
    float v0 = input[0];
    float v1 = (v0 + 1.0);
    return v1;
}"#;
    assert_eq!(result, expected);
}

#[test]
fn test_store_codegen() {
    let index = constant(0);
    let value = constant(42.0f32);
    let store_op = Store("output".to_string(), 10);
    // Since Store returns no value, we can't chain it.
    // The graph root is the Store node itself.
    let graph = Node::new(store_op, vec![index, value]);

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    // Note: The return value is currently incorrect because the generator
    // doesn't handle void functions yet. We are just testing the body.
    let expected = r#"void compute() {
    output[0] = 42.0;
}"#;
    assert_eq!(result, expected);
}