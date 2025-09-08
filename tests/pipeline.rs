use harp::{
    ast::DType,
    backend::{
        c::{CBuffer, CCompiler, CRenderer},
        Compiler, Kernel, Renderer,
    },
    graph::{shape::Expr, Graph},
    lowerer::Lowerer,
};

mod common;

#[test]
fn test_simple_add_pipeline() {
    common::setup();
    // 1. Build a simple computation graph (a + b)
    let mut graph = Graph::new();
    let a = graph.input(DType::F32, vec![Expr::from(4)]);
    let b = graph.input(DType::F32, vec![Expr::from(4)]);
    let c = &a + &b;
    graph.output(c);

    // 2. Lower the graph to an AST (Program)
    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&graph);

    // 3. Render the AST to C code
    let mut renderer = CRenderer::new();
    let c_code = renderer.render(program);

    // 4. Compile the C code into a kernel
    let mut compiler = CCompiler::new();
    assert!(compiler.is_available(), "C compiler not found");
    let signature = graph.signature();
    let mut kernel = compiler.compile(&c_code, signature);

    // 5. Prepare buffers and execute the kernel
    let input_a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let output_data: Vec<f32> = vec![0.0; 4];

    let buffer_a = CBuffer::from_slice(&input_a_data, &[4], DType::F32);
    let buffer_b = CBuffer::from_slice(&input_b_data, &[4], DType::F32);
    let buffer_c = CBuffer::from_slice(&output_data, &[4], DType::F32);

    // The order of buffers is inputs then outputs, as determined by the lowerer
    let buffers = vec![buffer_a, buffer_b, buffer_c];
    let shape_vars = vec![];

    let result_buffers = kernel.call(buffers, &shape_vars);

    // 6. Verify the result
    let result_vec = result_buffers[2].to_vec::<f32>();
    let expected_vec: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];

    assert_eq!(result_vec, expected_vec);
}

#[test]
fn test_dynamic_add_pipeline() {
    common::setup();
    // 1. Build graph with a dynamic shape variable "N"
    let mut graph = Graph::new();
    let n = graph.shape_var("N", 4); // Default value 4
    let a = graph.input(DType::F32, vec![n.clone()]);
    let b = graph.input(DType::F32, vec![n.clone()]);
    let c = &a + &b;
    graph.output(c);

    // 2. Lower, Render, Compile
    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(&graph);

    let mut renderer = CRenderer::new();
    let c_code = renderer.render(program);

    let mut compiler = CCompiler::new();
    assert!(compiler.is_available(), "C compiler not found");
    let signature = graph.signature();
    let mut kernel = compiler.compile(&c_code, signature);

    // 5. Prepare buffers and execute the kernel with a concrete value for "N"
    let n_val = 8;
    let input_a_data: Vec<f32> = (0..n_val).map(|i| i as f32).collect();
    let input_b_data: Vec<f32> = (0..n_val).map(|i| (i * 2) as f32).collect();
    let output_data: Vec<f32> = vec![0.0; n_val];

    let buffer_a = CBuffer::from_slice(&input_a_data, &[n_val], DType::F32);
    let buffer_b = CBuffer::from_slice(&input_b_data, &[n_val], DType::F32);
    let buffer_c = CBuffer::from_slice(&output_data, &[n_val], DType::F32);

    let buffers = vec![buffer_a, buffer_b, buffer_c];
    let shape_vars = vec![n_val]; // Pass the concrete value for "N"

    let result_buffers = kernel.call(buffers, &shape_vars);

    // 6. Verify the result
    let result_vec = result_buffers[2].to_vec::<f32>();
    let expected_vec: Vec<f32> = (0..n_val).map(|i| (i + i * 2) as f32).collect();

    assert_eq!(result_vec, expected_vec);
}
