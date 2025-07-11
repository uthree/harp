use harp::codegen::CodeGenerator;
use harp::c_renderer::CRenderer;
use harp::node::{self, constant, variable};

#[test]
fn test_simple_graph_rendering() {
    let a = variable("a");
    let b = constant(2.0f32);
    let graph = a + b;

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    assert_eq!(result, "(a + 2.0)");
}

#[test]
fn test_fused_op_rendering() {
    let a = variable("a");
    let b = variable("b");
    let graph = a - b; // OpSub is a FusedOp

    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let result = codegen.generate(&graph);

    // OpSub should be expanded to (a + (b * -1.0))
    assert_eq!(result, "(a + (b * -1.0))");
}
