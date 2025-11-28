use super::super::*;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_fused_elementwise() {
    use crate::ast::helper::wildcard;

    // (a + b) * c を融合ノードとして生成
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![10]);
    let b = graph.input("b", GraphDType::F32, vec![10]);
    let c = graph.input("c", GraphDType::F32, vec![10]);

    // 融合演算を定義: (Wildcard("0") + Wildcard("1")) * Wildcard("2")
    let expr = (wildcard("0") + wildcard("1")) * wildcard("2");

    let result = crate::graph::ops::fused_elementwise(vec![a, b, c], expr);

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
    use crate::ast::helper::wildcard;
    use crate::graph::ops::ReduceOp;

    // reduce_sum(a * b, axis=0) を融合ノードとして生成
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![10, 20]);
    let b = graph.input("b", GraphDType::F32, vec![10, 20]);

    // 融合演算を定義: Wildcard("0") * Wildcard("1")
    let expr = wildcard("0") * wildcard("1");

    let result = crate::graph::ops::fused_elementwise_reduce(vec![a, b], expr, ReduceOp::Sum, 0);

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

#[test]
fn test_lower_fused_elementwise_cumulative() {
    use crate::ast::helper::wildcard;
    use crate::graph::ops::CumulativeOp;

    // cumsum(x^2) を融合ノードとして生成
    let mut graph = Graph::new();
    let x = graph.input("x", GraphDType::F32, vec![4, 8]);

    // 融合演算を定義: Wildcard("0") * Wildcard("0") (二乗)
    let expr = wildcard("0") * wildcard("0");

    let result = crate::graph::ops::fused_elementwise_cumulative(
        vec![x],
        expr,
        CumulativeOp::Sum,
        1, // 軸1に沿って累積
    );

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output
    use crate::ast::AstNode;
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "output");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c_like::CLikeRenderer;
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_fused_elementwise_cumulative ===\n{}\n",
        code
    );
}
