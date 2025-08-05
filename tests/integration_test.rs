use harp::{
    ast::DType,
    backend::{Buffer, Compiler, Kernel, Renderer, c::*},
    tensor::{graph::Graph, lowerer::Lowerer},
};
use std::ffi::c_void;

// A mock buffer for testing that holds a simple Vec<f32>.
#[derive(Debug, PartialEq)]
struct MockBuffer(Vec<f32>);

impl Buffer for MockBuffer {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.0.as_mut_ptr() as *mut c_void
    }
}

#[test]
fn test_full_flow_add() {
    // 1. Build Graph
    let graph = Graph::new();
    let a = graph.input(DType::F32, vec![4.into()]);
    let b = graph.input(DType::F32, vec![4.into()]);
    let c = (a + b).as_output();

    // 2. Lower to AST
    let mut lowerer = Lowerer::new(&graph);
    let ast = lowerer.lower();

    // 3. Render to C code
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 4. Compile C code
    let mut compiler = CCompiler::default();
    assert!(compiler.check_availability());
    let kernel = <CCompiler as Compiler<MockBuffer, _, ()>>::compile(&mut compiler, &code);

    // 5. Prepare data and run kernel
    let buf_a = MockBuffer(vec![1.0, 2.0, 3.0, 4.0]);
    let buf_b = MockBuffer(vec![5.0, 6.0, 7.0, 8.0]);
    // Output buffer, initialized to 0. The kernel will write the result here.
    let buf_c = MockBuffer(vec![0.0; 4]);

    // The lowerer assigns buffers in order of discovery: a, b, then c.
    let buffers = vec![buf_a, buf_b, buf_c];
    let result_buffers = kernel.call(buffers, vec![]);

    // 6. Verify results
    let expected = MockBuffer(vec![6.0, 8.0, 10.0, 12.0]);
    // The output buffer `c` is the third one in the list.
    assert_eq!(result_buffers[2], expected);
}
