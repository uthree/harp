// tests/c_backend.rs

use harp::ast::{AstNode, DType};
use harp::backend::c::{CBackend, CBuffer, CCompiler, CRenderer};
use harp::backend::{Backend, Compiler, Kernel, Renderer, TryIntoNdarray};
use harp::graph::lowerer::Lowerer;
use harp::graph::Graph;
use ndarray::ArrayD;
use std::ffi::c_void;

use std::f32;

// Helper function to render an AST node and compare it with the expected output.
fn assert_render(node: AstNode, expected: &str) {
    let mut renderer = CRenderer::new();
    let rendered_code = renderer.render(node);
    // Normalize whitespace and remove initial headers for easier comparison
    let cleaned_code = rendered_code
        .lines()
        .skip(3) // Skip header lines
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
    // F32
    let ast: AstNode = f32::consts::PI.into();
    let mut renderer = CRenderer::new();
    let rendered_code = renderer.render(ast);
    let cleaned_code = rendered_code
        .lines()
        .skip(3) // Skip header lines
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<String>();
    assert!(cleaned_code.starts_with("3.1415927"));

    // I8
    let ast: AstNode = (42i8).into();
    let rendered_code = renderer.render(ast);
    let cleaned_code = rendered_code
        .lines()
        .skip(3)
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<String>();
    assert_eq!(cleaned_code, "(int8_t)42");

    // U32
    let ast: AstNode = (123u32).into();
    let rendered_code = renderer.render(ast);
    let cleaned_code = rendered_code
        .lines()
        .skip(3)
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<String>();
    assert_eq!(cleaned_code, "123u");

    // I64
    let ast: AstNode = (9999999999i64).into();
    let rendered_code = renderer.render(ast);
    let cleaned_code = rendered_code
        .lines()
        .skip(3)
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<String>();
    assert_eq!(cleaned_code, "9999999999ll");
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

/// Helper function to create a CBuffer from a slice of data.
fn buffer_from_slice<T: Clone>(data: &[T], shape: &[usize], dtype: DType) -> CBuffer {
    assert_eq!(
        data.len(),
        shape.iter().product(),
        "Data length must match the product of shape dimensions"
    );
    let byte_size = std::mem::size_of_val(data);
    let ptr = unsafe { libc::malloc(byte_size) };
    assert!(!ptr.is_null(), "Failed to allocate memory for buffer");
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, byte_size);
    }
    CBuffer {
        ptr: ptr as *mut c_void,
        shape: shape.to_vec(),
        dtype,
    }
}

/// Helper function to create an empty CBuffer for output.
fn empty_buffer(shape: &[usize], dtype: DType) -> CBuffer {
    let size: usize = shape.iter().product();
    let byte_size = size * dtype.size_in_bytes();
    let ptr = unsafe { libc::malloc(byte_size) };
    assert!(!ptr.is_null(), "Failed to allocate memory for buffer");
    CBuffer {
        ptr: ptr as *mut c_void,
        shape: shape.to_vec(),
        dtype,
    }
}

#[test]
fn test_c_backend_e2e_add() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    // 1. Build Graph: c = a + b
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a + b).as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();

    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);
    let c_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer, c_buffer], &[]);

    // 5. Verify results
    let c_result_array = result_buffers[2].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(c_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_neg() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    // 1. Build Graph: b = -a
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (-a).as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);

    // 5. Verify results
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data.iter().map(|&x| -x).collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_rem() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    // 1. Build Graph: c = a % b
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a % b).as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| (i + 1) as f32).collect();

    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);
    let c_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer, c_buffer], &[]);

    // 5. Verify results
    let c_result_array = result_buffers[2].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x % y)
        .collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(c_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_lt() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    // 1. Build Graph: c = a < b
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    a.lt(b).as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|_i| 5.0f32).collect();

    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);
    let c_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer, c_buffer], &[]);

    // 5. Verify results
    let c_result_array = result_buffers[2].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| if x < y { 1.0 } else { 0.0 })
        .collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(c_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_sin() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    // 1. Build Graph: b = sin(a)
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    a.sin().as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);

    // 5. Verify results
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data.iter().map(|&x| x.sin()).collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_sqrt() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    a.sqrt().as_output();

    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);
    let kernel = compiler.compile(&code, details);

    let a_data: Vec<f32> = (0..10).map(|i| (i * i) as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data.iter().map(|&x| x.sqrt()).collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_log2() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    a.log2().as_output();

    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);
    let kernel = compiler.compile(&code, details);

    let a_data: Vec<f32> = (1..11).map(|i| i as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data.iter().map(|&x| x.log2()).collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_exp2() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    a.exp2().as_output();

    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);
    let kernel = compiler.compile(&code, details);

    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data.iter().map(|&x| x.exp2()).collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_recip() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    a.recip().as_output();

    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);
    let kernel = compiler.compile(&code, details);

    let a_data: Vec<f32> = (1..11).map(|i| i as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data.iter().map(|&x| 1.0 / x).collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_reduce_sum() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    // 1. Build Graph: b = a.sum(axis=1)
    let graph = Graph::new();
    let input_shape = vec![2, 3];
    let output_shape = vec![2];
    let a = graph.input(DType::F32, input_shape.iter().map(|&d| d.into()).collect());
    a.sum(1).as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_buffer = buffer_from_slice(&a_data, &input_shape, DType::F32);
    let b_buffer = empty_buffer(&output_shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);

    // 5. Verify results
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();
    let expected_data: Vec<f32> = vec![6.0, 15.0];
    let expected_array = ArrayD::from_shape_vec(output_shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_reduce_max() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    let graph = Graph::new();
    let input_shape = vec![2, 3];
    let output_shape = vec![2];
    let a = graph.input(DType::F32, input_shape.iter().map(|&d| d.into()).collect());
    a.max(1).as_output();

    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);
    let kernel = compiler.compile(&code, details);

    let a_data: Vec<f32> = vec![1.0, 5.0, 2.0, 8.0, 3.0, 4.0];
    let a_buffer = buffer_from_slice(&a_data, &input_shape, DType::F32);
    let b_buffer = empty_buffer(&output_shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);

    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();
    let expected_data: Vec<f32> = vec![5.0, 8.0];
    let expected_array = ArrayD::from_shape_vec(output_shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_reduce_prod() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    let graph = Graph::new();
    let input_shape = vec![2, 3];
    let output_shape = vec![2];
    let a = graph.input(DType::F32, input_shape.iter().map(|&d| d.into()).collect());
    a.prod(1).as_output();

    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);
    let kernel = compiler.compile(&code, details);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_buffer = buffer_from_slice(&a_data, &input_shape, DType::F32);
    let b_buffer = empty_buffer(&output_shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);

    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();
    let expected_data: Vec<f32> = vec![6.0, 120.0];
    let expected_array = ArrayD::from_shape_vec(output_shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_cumulative_sum() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    // 1. Build Graph: b = a.cumsum(axis=1)
    let graph = Graph::new();
    let shape = vec![2, 3];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    a.cumsum(1).as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = empty_buffer(&shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, b_buffer], &[]);

    // 5. Verify results
    let b_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();
    let expected_data: Vec<f32> = vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0];
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(b_result_array, expected_array);
}

#[test]
fn test_c_backend_e2e_slice() {
    harp::init_logger();
    let mut compiler = CCompiler::new();
    if !compiler.is_available() {
        return;
    }

    // 1. Build Graph: b = a[1:3, 2:4].sin()
    let graph = Graph::new();
    let input_shape = vec![4, 5];
    let sliced_shape = vec![2, 2];
    let a = graph.input(DType::F32, input_shape.iter().map(|&d| d.into()).collect());
    let b = a.slice(vec![(1.into(), 3.into()), (2.into(), 4.into())]);
    b.sin().as_output();

    // 2. Lower and Render
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 3. Compile
    let kernel = compiler.compile(&code, details);

    // 4. Prepare data and call kernel
    let a_data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let a_buffer = buffer_from_slice(&a_data, &input_shape, DType::F32);
    let c_buffer = empty_buffer(&sliced_shape, DType::F32);

    let mut result_buffers = kernel.call(vec![a_buffer, c_buffer], &[]);

    // 5. Verify results
    let c_result_array = result_buffers[1].try_into_ndarray::<f32>().unwrap();

    // Manually slice and apply sin
    let a_ndarray = ArrayD::from_shape_vec(input_shape, a_data).unwrap();
    let sliced_view = a_ndarray.slice(ndarray::s![1..3, 2..4]);
    let expected_array = sliced_view.mapv(f32::sin).into_dyn();

    assert_eq!(c_result_array, expected_array);
}

#[test]
fn test_cbackend_call_simple_add() {
    harp::init_logger();
    let mut backend = CBackend::new();
    if !backend.is_available() {
        eprintln!("Skipping C backend E2E test: C compiler not found.");
        return;
    }

    // 1. Build Graph: c = a + b
    let graph = Graph::new();
    let shape = vec![10];
    let a = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    let b = graph.input(DType::F32, shape.iter().map(|&d| d.into()).collect());
    (a + b).as_output();

    // 2. Prepare input data
    let a_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();

    let a_buffer = buffer_from_slice(&a_data, &shape, DType::F32);
    let b_buffer = buffer_from_slice(&b_data, &shape, DType::F32);

    // 3. Call the backend
    let mut result_buffer = backend.call(graph, vec![a_buffer, b_buffer], vec![]);

    // 4. Verify results
    let result_array = result_buffer.try_into_ndarray::<f32>().unwrap();

    let expected_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let expected_array = ArrayD::from_shape_vec(shape, expected_data).unwrap();

    assert_eq!(result_array, expected_array);
}
