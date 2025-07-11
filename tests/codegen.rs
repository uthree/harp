use harp::backend::c::CRenderer;
use harp::backend::codegen::CodeGenerator;
use harp::backend::renderer::Renderer;
use harp::node::{constant, variable, Node};
use harp::op::{Input, Load, Loop, LoopVariable, Store};

fn generate_code(graph: &Node, args: &[(&str, &str)], return_type: &str) -> String {
    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let instructions = codegen.generate(graph);
    renderer.render_function("test_fn", args, &instructions, return_type)
}

#[test]
fn test_simple_codegen() {
    let a = variable("a");
    let b = constant(2.0f32);
    let graph = a + b;
    let code = generate_code(&graph, &[("float", "a")], "float");

    let expected = r#"float test_fn(float a) {
    float v0 = (a + 2.0);
    return v0;
}"#;
    assert_eq!(code.replace_whitespace(), expected.replace_whitespace());
}

#[test]
fn test_fused_op_codegen() {
    let a = variable("a");
    let b = variable("b");
    let graph = a - b; // OpSub is a FusedOp
    let code = generate_code(&graph, &[("float", "a"), ("float", "b")], "float");

    let expected = r#"float test_fn(float a, float b) {
    float v0 = (b * -1.0);
    float v1 = (a + v0);
    return v1;
}"#;
    assert_eq!(code.replace_whitespace(), expected.replace_whitespace());
}

#[test]
fn test_loop_codegen() {
    let i = Node::new(LoopVariable, vec![]);
    let c = Node::new(Input("c".to_string()), vec![]);
    let store_node = Node::new(Store, vec![c.clone(), i.clone(), i]);
    let count = constant(10);
    let graph = Node::new(
        Loop {
            count: count.clone(),
            body: store_node,
        },
        vec![count, c],
    );
    let code = generate_code(&graph, &[("float*", "c")], "void");

    let expected = r#"void test_fn(float* c) {
    for (int i = 0; i < 10; ++i) {
        c[i] = i;
    }
}"#;
    assert_eq!(code.replace_whitespace(), expected.replace_whitespace());
}

#[test]
fn test_load_store_codegen() {
    let i = Node::new(LoopVariable, vec![]);
    let a = Node::new(Input("a".to_string()), vec![]);
    let b = Node::new(Input("b".to_string()), vec![]);
    let c = Node::new(Input("c".to_string()), vec![]);

    let a_i = Node::new(Load, vec![a.clone(), i.clone()]);
    let b_i = Node::new(Load, vec![b.clone(), i.clone()]);
    let add_result = a_i + b_i;
    let store_node = Node::new(Store, vec![c.clone(), i, add_result]);
    let count = constant(10);
    let graph = Node::new(
        Loop {
            count: count.clone(),
            body: store_node,
        },
        vec![count, a, b, c],
    );
    let code = generate_code(
        &graph,
        &[
            ("const float*", "a"),
            ("const float*", "b"),
            ("float*", "c"),
        ],
        "void",
    );

    let expected = r#"void test_fn(const float* a, const float* b, float* c) {
    for (int i = 0; i < 10; ++i) {
        float v0 = a[i];
        float v1 = b[i];
        float v2 = (v0 + v1);
        c[i] = v2;
    }
}"#;
    assert_eq!(code.replace_whitespace(), expected.replace_whitespace());
}

// Helper to normalize whitespace for reliable comparisons
trait StringWhitespace {
    fn replace_whitespace(&self) -> String;
}

impl StringWhitespace for str {
    fn replace_whitespace(&self) -> String {
        self.split_whitespace().collect::<Vec<&str>>().join(" ")
    }
}