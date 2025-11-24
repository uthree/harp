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
fn test_conv1d_backward_stride_2() {
    let _ = env_logger::builder().is_test(true).try_init();

    // stride=2の場合のConv1d backward
    // Input: (C_in=1, L=5), Kernel: (C_out=1, C_in=1, k=3)
    // Output: (1, 2) with stride=2
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

    let output = x.conv1d(&kernel, 2, 1, 1); // stride=2

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("Input gradient (stride=2): {:?}", x.grad());
    eprintln!("Kernel gradient (stride=2): {:?}", kernel.grad());
}

#[test]
fn test_conv2d_backward_simple() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Conv2d backward (stride対応)
    // Input: (C_in=1, H=4, W=4), Kernel: (C_out=1, C_in=1, kh=2, kw=2)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(4),
                crate::graph::shape::Expr::from(4),
            ]),
        true, // requires_grad
    );

    let kernel = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(0.5f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
            ]),
        true, // requires_grad
    );

    let output = x.conv2d(&kernel, (1, 1), (1, 1), 1);

    let scalar = output.sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("Conv2d Input gradient: {:?}", x.grad());
    eprintln!("Conv2d Kernel gradient: {:?}", kernel.grad());
}

#[test]
fn test_conv3d_backward_simple() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Conv3d backward (stride対応)
    // Input: (C_in=1, D=3, H=3, W=3), Kernel: (C_out=1, C_in=1, kD=2, kH=2, kW=2)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(3),
                crate::graph::shape::Expr::from(3),
                crate::graph::shape::Expr::from(3),
            ]),
        true, // requires_grad
    );

    let kernel = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(0.5f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
            ]),
        true, // requires_grad
    );

    let output = x.conv3d(&kernel, (1, 1, 1), (1, 1, 1), 1);

    let scalar = output.sum(0).sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("Conv3d Input gradient: {:?}", x.grad());
    eprintln!("Conv3d Kernel gradient: {:?}", kernel.grad());
}
