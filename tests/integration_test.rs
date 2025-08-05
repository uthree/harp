use harp::{
    ast::{DType, Expr},
    backend::{Buffer, Compiler, Kernel, Renderer, c::*},
    tensor::{graph::Graph, lowerer::Lowerer},
};
use std::ffi::c_void;

// A mock buffer for testing that holds a simple Vec<f32>.
#[derive(Debug, PartialEq)]
struct MockBuffer {
    data: Vec<f32>,
    dtype: DType,
    shape: Vec<Expr>,
}

impl Buffer for MockBuffer {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.data.as_mut_ptr() as *mut c_void
    }
    fn dtype(&self) -> DType {
        self.dtype.clone()
    }
    fn shape(&self) -> &[Expr] {
        &self.shape
    }
}

#[test]
fn test_full_flow_add() {
    // 1. Build Graph
    let graph = Graph::new();
    let shape: Vec<Expr> = vec![4.into()];
    let a = graph.input(DType::F32, shape.clone());
    let b = graph.input(DType::F32, shape.clone());
    let _c = (a + b).as_output();

    // 2. Lower to AST and get KernelDetails
    let mut lowerer = Lowerer::new(&graph);
    let (ast, details) = lowerer.lower();

    // -- Verify KernelDetails --
    assert_eq!(details.buffers.len(), 3);
    assert_eq!(details.shape_variables.len(), 0);
    // Input A
    assert_eq!(details.buffers[0].dtype, DType::F32);
    assert_eq!(details.buffers[0].shape, shape);
    // Input B
    assert_eq!(details.buffers[1].dtype, DType::F32);
    assert_eq!(details.buffers[1].shape, shape);
    // Output C
    assert_eq!(details.buffers[2].dtype, DType::F32);
    assert_eq!(details.buffers[2].shape, shape);

    // 3. Render to C code
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 4. Compile C code
    let mut compiler = CCompiler::default();
    assert!(compiler.check_availability());
    let kernel = <CCompiler as Compiler<MockBuffer, _, ()>>::compile(&mut compiler, &code);

    // 5. Prepare data and run kernel
    let buf_a = MockBuffer {
        data: vec![1.0, 2.0, 3.0, 4.0],
        dtype: DType::F32,
        shape: shape.clone(),
    };
    let buf_b = MockBuffer {
        data: vec![5.0, 6.0, 7.0, 8.0],
        dtype: DType::F32,
        shape: shape.clone(),
    };
    // Output buffer, initialized to 0.
    let buf_c = MockBuffer {
        data: vec![0.0; 4],
        dtype: DType::F32,
        shape: shape.clone(),
    };

    // The lowerer assigns buffers in order: a, b, then c.
    let buffers = vec![buf_a, buf_b, buf_c];
    let result_buffers = kernel.call(buffers, vec![]);

    // 6. Verify results
    let expected = MockBuffer {
        data: vec![6.0, 8.0, 10.0, 12.0],
        dtype: DType::F32,
        shape: shape.clone(),
    };
    // The output buffer `c` is the third one in the list.
    assert_eq!(result_buffers[2], expected);
}
