/// Conv演算の勾配計算のテスト
///
/// 注意: 現在はconv1dのみ実装されており、stride=1, dilation=1, groups=1のみサポート
use crate::autograd::Tensor;

#[test]
fn test_conv1d_backward_simple() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 簡単なconv1d + 勾配計算のテスト
    // Input: (C_in=1, L=5), Kernel: (C_out=1, C_in=1, k=3)
    // Output: (1, 3)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(5),
            ]),
        true, // requires_grad
    );

    let kernel = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(0.5f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(3),
            ]),
        true, // requires_grad
    );

    let output = x.conv1d(&kernel, 1, 1, 1);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0);

    // This should NOT panic (backward is now implemented)
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("Input gradient: {:?}", x.grad());
    eprintln!("Kernel gradient: {:?}", kernel.grad());
}

#[test]
#[ignore = "stride != 1のbackward実装は後回し"]
fn test_conv1d_backward_stride_not_supported() {
    // stride != 1 はサポートされていないことを確認
    let x = Tensor::ones(vec![1, 5]);
    let kernel = Tensor::ones(vec![1, 1, 3]);

    let output = x.conv1d(&kernel, 2, 1, 1); // stride=2

    let _scalar = output.sum(0).sum(0);
    // TODO: この場合のbackwardを実装
    // _scalar.backward();
}

#[test]
#[ignore = "Conv2dのbackward実装は後回し"]
fn test_conv2d_backward_not_implemented() {
    // Conv2dの勾配はまだ未実装
    let x = Tensor::ones(vec![1, 4, 4]);
    let kernel = Tensor::ones(vec![1, 1, 2, 2]);

    let output = x.conv2d(&kernel, (1, 1), (1, 1), 1);

    let _scalar = output.sum(0).sum(0).sum(0);
    // TODO: Conv2dのbackwardを実装
    // scalar.backward();
}

#[test]
#[ignore = "Conv3dのbackward実装は後回し"]
fn test_conv3d_backward_not_implemented() {
    // Conv3dの勾配はまだ未実装
    let x = Tensor::ones(vec![1, 3, 3, 3]);
    let kernel = Tensor::ones(vec![1, 1, 2, 2, 2]);

    let output = x.conv3d(&kernel, (1, 1, 1), (1, 1, 1), 1);

    let _scalar = output.sum(0).sum(0).sum(0).sum(0);
    // TODO: Conv3dのbackwardを実装
    // scalar.backward();
}
