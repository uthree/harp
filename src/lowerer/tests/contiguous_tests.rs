use super::super::*;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_contiguous_2d() {
    let _ = env_logger::builder().is_test(true).try_init();

    use crate::graph::ops::GraphOp;
    use crate::graph::shape::View;

    // 2次元テンソルの転置を持つノードを作成
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![3, 4]) // 3x4の行列
        .build();

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

    // パラメータをチェック: input0, output, shape0, shape1
    assert_eq!(function.params.len(), 4);
    assert_eq!(function.params[0].name, "input0");
    assert_eq!(function.params[1].name, "output");
    assert_eq!(function.params[2].name, "shape0");
    assert_eq!(function.params[3].name, "shape1");

    // 生成されたコードを表示
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("contiguous_2d_kernel", &function);
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
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();

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

    // パラメータをチェック: input0, output, shape0
    assert_eq!(function.params.len(), 3);
    assert_eq!(function.params[0].name, "input0");
    assert_eq!(function.params[1].name, "output");
    assert_eq!(function.params[2].name, "shape0");

    // 生成されたコードを表示
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("contiguous_1d_kernel", &function);
    eprintln!(
        "\n=== Generated Code for test_lower_contiguous_1d ===\n{}\n",
        code
    );
}
