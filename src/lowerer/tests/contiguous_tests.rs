use super::super::*;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_contiguous_2d() {
    let _ = env_logger::builder().is_test(true).try_init();

    use crate::graph::ops::GraphOp;
    use crate::graph::shape::View;

    // 2次元テンソルの転置を持つノードを作成
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![3, 4]); // 3x4の行列

    // 転置されたView（4x3になる）
    let transposed_view = a.view.clone().permute(vec![1, 0]);

    // Viewノードを作成（転置操作）
    let view_node = GraphNode::new(
        a.dtype.clone(),
        GraphOp::View(transposed_view.clone()),
        vec![a.clone()],
        transposed_view.clone(),
    );

    // Contiguousノードを作成（実際のメモリレイアウト変換）
    let contiguous_node = GraphNode::new(
        view_node.dtype.clone(),
        GraphOp::Contiguous {
            elementwise_strategies: None,
        },
        vec![view_node.clone()],
        View::contiguous(transposed_view.shape().to_vec()),
    );

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&contiguous_node, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output (shapeは定数なのでパラメータ不要)
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
        "\n=== Generated Code for test_lower_contiguous_2d ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_contiguous_1d() {
    let _ = env_logger::builder().is_test(true).try_init();

    use crate::graph::ops::GraphOp;
    use crate::graph::shape::View;

    // 1次元テンソルのflip（反転）を持つノードを作成
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::F32, vec![10]);

    // 反転されたView
    let flipped_view = a.view.clone().flip(0);

    // Viewノードを作成（反転操作）
    let view_node = GraphNode::new(
        a.dtype.clone(),
        GraphOp::View(flipped_view.clone()),
        vec![a.clone()],
        flipped_view.clone(),
    );

    // Contiguousノードを作成（実際のメモリレイアウト変換）
    let contiguous_node = GraphNode::new(
        view_node.dtype.clone(),
        GraphOp::Contiguous {
            elementwise_strategies: None,
        },
        vec![view_node.clone()],
        View::contiguous(flipped_view.shape().to_vec()),
    );

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&contiguous_node, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output (shapeは定数なのでパラメータ不要)
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
        "\n=== Generated Code for test_lower_contiguous_1d ===\n{}\n",
        code
    );
}
