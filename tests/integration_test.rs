use harp::{
    ast::DType,
    backend::{Buffer, Compiler, Kernel, Renderer, c::*},
    opt::graph::{GraphRewriter, get_fusion_rules},
    tensor::{graph::Graph, lowerer::Lowerer, shape::expr::Expr},
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
    fn shape(&self) -> Vec<Expr> {
        self.shape.clone()
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

#[test]
fn test_full_flow_complex_fusion() {
    // 1. Build Graph for (a * b) + c
    let graph = Graph::new();
    let shape: Vec<Expr> = vec![4.into()];
    let a = graph.input(DType::F32, shape.clone());
    let b = graph.input(DType::F32, shape.clone());
    let c = graph.input(DType::F32, shape.clone());
    let _d = ((a * b) + c).as_output();
    assert_eq!(graph.nodes.borrow().len(), 5); // a, b, c, mul, add

    // 2. Optimize graph
    let rules = get_fusion_rules();
    let rewriter = GraphRewriter::new(rules);
    let optimized_graph = rewriter.apply(&graph);
    assert_eq!(optimized_graph.nodes.borrow().len(), 4); // a, b, c, fused_op

    // 3. Lower to AST and get KernelDetails
    let mut lowerer = Lowerer::new(&optimized_graph);
    let (ast, details) = lowerer.lower();

    // -- Verify KernelDetails --
    assert_eq!(details.buffers.len(), 4); // a, b, c, d
    assert_eq!(details.shape_variables.len(), 0);

    // 4. Render to C code
    let mut renderer = CRenderer::new();
    let code = renderer.render(ast);

    // 5. Compile C code
    let mut compiler = CCompiler::default();
    assert!(compiler.check_availability());
    let kernel = <CCompiler as Compiler<MockBuffer, _, ()>>::compile(&mut compiler, &code);

    // 6. Prepare data and run kernel
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
    let buf_c = MockBuffer {
        data: vec![9.0, 10.0, 11.0, 12.0],
        dtype: DType::F32,
        shape: shape.clone(),
    };
    // Output buffer, initialized to 0.
    let buf_d = MockBuffer {
        data: vec![0.0; 4],
        dtype: DType::F32,
        shape: shape.clone(),
    };

    // The lowerer assigns buffers in order: a, b, c, then d.
    let buffers = vec![buf_a, buf_b, buf_c, buf_d];
    let result_buffers = kernel.call(buffers, vec![]);

    // 7. Verify results
    let expected = MockBuffer {
        // (1*5)+9=14, (2*6)+10=22, (3*7)+11=32, (4*8)+12=44
        data: vec![14.0, 22.0, 32.0, 44.0],
        dtype: DType::F32,
        shape: shape.clone(),
    };
    // The output buffer `d` is the fourth one in the list.
    assert_eq!(result_buffers[3], expected);
}
