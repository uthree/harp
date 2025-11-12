use super::super::*;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_fused_elementwise() {
    use crate::graph::ops::{ElementwiseOp, FusedElementwiseOp, FusedInput};

    // (a + b) * c を融合ノードとして生成
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let c = graph
        .input("c")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();

    // 融合演算を定義
    let ops = vec![
        FusedElementwiseOp {
            op: ElementwiseOp::Add,
            inputs: vec![FusedInput::GraphInput(0), FusedInput::GraphInput(1)],
        },
        FusedElementwiseOp {
            op: ElementwiseOp::Mul,
            inputs: vec![FusedInput::IntermediateResult(0), FusedInput::GraphInput(2)],
        },
    ];

    let result = crate::graph::ops::fused_elementwise(vec![a, b, c], ops);

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, input1, input2, output (shapeは定数なのでパラメータ不要)
    use crate::ast::AstNode;
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "input1");
        assert_eq!(params[2].name, "input2");
        assert_eq!(params[3].name, "output");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c_like::CLikeRenderer;
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_fused_elementwise ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_fused_elementwise_reduce() {
    use crate::graph::ops::{ElementwiseOp, FusedElementwiseOp, FusedInput, ReduceOp};

    // reduce_sum(a * b, axis=0) を融合ノードとして生成
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10, 20])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10, 20])
        .build();

    // 融合演算を定義: a * b
    let ops = vec![FusedElementwiseOp {
        op: ElementwiseOp::Mul,
        inputs: vec![FusedInput::GraphInput(0), FusedInput::GraphInput(1)],
    }];

    let result = crate::graph::ops::fused_elementwise_reduce(vec![a, b], ops, ReduceOp::Add, 0);

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, input1, output (shapeは定数なのでパラメータ不要)
    use crate::ast::AstNode;
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "input1");
        assert_eq!(params[2].name, "output");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c_like::CLikeRenderer;
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_fused_elementwise_reduce ===\n{}\n",
        code
    );
}
