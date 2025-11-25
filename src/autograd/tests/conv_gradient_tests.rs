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

#[test]
fn test_conv1d_backward_groups() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Group Conv1d backward (groups=2)
    // Input: (C_in=2, L=5), Kernel: (C_out=2, C_in/groups=1, k=3)
    // groups=2なので、depthwise convolution
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(5),
            ]),
        true, // requires_grad
    );

    // Kernelはcontiguousである必要があるため、expandではなく直接作成
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 1, 3])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    let output = x.conv1d(&kernel, 1, 1, 2); // groups=2

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

    eprintln!("Conv1d (groups=2) Input gradient: {:?}", x.grad());
    eprintln!("Conv1d (groups=2) Kernel gradient: {:?}", kernel.grad());
}

#[test]
fn test_conv2d_backward_groups() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Group Conv2d backward (groups=2)
    // Input: (C_in=2, H=4, W=4), Kernel: (C_out=2, C_in/groups=1, kh=2, kw=2)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(4),
                crate::graph::shape::Expr::from(4),
            ]),
        true, // requires_grad
    );

    // Kernelはcontiguousである必要があるため、expandではなく直接作成
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 1, 2, 2])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    let output = x.conv2d(&kernel, (1, 1), (1, 1), 2); // groups=2

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("Conv2d (groups=2) Input gradient: {:?}", x.grad());
    eprintln!("Conv2d (groups=2) Kernel gradient: {:?}", kernel.grad());
}

#[test]
fn test_conv2d_backward_groups_stride() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Group Conv2d backward with stride (groups=2, stride=(2,2))
    // Input: (C_in=4, H=6, W=6), Kernel: (C_out=4, C_in/groups=2, kh=3, kw=3)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(4),
                crate::graph::shape::Expr::from(6),
                crate::graph::shape::Expr::from(6),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_out=4, C_in/groups=2, kH=3, kW=3)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![4, 2, 3, 3])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // stride=(2,2), groups=2
    let output = x.conv2d(&kernel, (2, 2), (1, 1), 2);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("Conv2d (groups=2, stride=2) Input gradient: {:?}", x.grad());
    eprintln!(
        "Conv2d (groups=2, stride=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}

#[test]
fn test_conv2d_backward_groups_dilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Group Conv2d backward with dilation (groups=2, dilation=(2,2))
    // Input: (C_in=4, H=8, W=8), Kernel: (C_out=4, C_in/groups=2, kh=3, kw=3)
    // effective kernel size = 3 + (3-1)*(2-1) = 5
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(4),
                crate::graph::shape::Expr::from(8),
                crate::graph::shape::Expr::from(8),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_out=4, C_in/groups=2, kH=3, kW=3)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![4, 2, 3, 3])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // dilation=(2,2), groups=2
    let output = x.conv2d(&kernel, (1, 1), (2, 2), 2);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!(
        "Conv2d (groups=2, dilation=2) Input gradient: {:?}",
        x.grad()
    );
    eprintln!(
        "Conv2d (groups=2, dilation=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}

#[test]
fn test_conv_transpose2d_backward_simple() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose2d backward (simple case)
    // Input: (C_in=2, H=3, W=3), Kernel: (C_in=2, C_out=4, kH=2, kW=2)
    // Output with stride=1: (4, 4, 4)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(3),
                crate::graph::shape::Expr::from(3),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=2, C_out=4, kH=2, kW=2)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 4, 2, 2])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // stride=(1,1), padding=(0,0), output_padding=(0,0), dilation=(1,1), groups=1
    let output = x.conv_transpose2d(&kernel, (1, 1), (0, 0), (0, 0), (1, 1), 1);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("ConvTranspose2d Input gradient: {:?}", x.grad());
    eprintln!("ConvTranspose2d Kernel gradient: {:?}", kernel.grad());
}

#[test]
fn test_conv_transpose2d_backward_stride() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose2d backward with stride
    // Input: (C_in=2, H=3, W=3), Kernel: (C_in=2, C_out=4, kH=3, kW=3)
    // Output with stride=2: (4, 7, 7)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(3),
                crate::graph::shape::Expr::from(3),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=2, C_out=4, kH=3, kW=3)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 4, 3, 3])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // stride=(2,2)
    let output = x.conv_transpose2d(&kernel, (2, 2), (0, 0), (0, 0), (1, 1), 1);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("ConvTranspose2d (stride=2) Input gradient: {:?}", x.grad());
    eprintln!(
        "ConvTranspose2d (stride=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}

#[test]
fn test_conv_transpose2d_backward_groups() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose2d backward with groups
    // Input: (C_in=4, H=3, W=3), Kernel: (C_in=4, C_out/groups=2, kH=2, kW=2)
    // groups=2, so C_out=4
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(4),
                crate::graph::shape::Expr::from(3),
                crate::graph::shape::Expr::from(3),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=4, C_out/groups=2, kH=2, kW=2)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![4, 2, 2, 2])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // groups=2
    let output = x.conv_transpose2d(&kernel, (1, 1), (0, 0), (0, 0), (1, 1), 2);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("ConvTranspose2d (groups=2) Input gradient: {:?}", x.grad());
    eprintln!(
        "ConvTranspose2d (groups=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}

#[test]
fn test_conv_transpose1d_backward_simple() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose1d backward (simple case)
    // Input: (C_in=2, L=4), Kernel: (C_in=2, C_out=3, k=3)
    // Output with stride=1: (3, 6)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(4),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=2, C_out=3, k=3)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 3, 3])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // stride=1
    let output = x.conv_transpose1d(&kernel, 1, 0, 0, 1, 1);

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

    eprintln!("ConvTranspose1d Input gradient: {:?}", x.grad());
    eprintln!("ConvTranspose1d Kernel gradient: {:?}", kernel.grad());
}

#[test]
fn test_conv_transpose1d_backward_stride() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose1d backward with stride
    // Input: (C_in=2, L=4), Kernel: (C_in=2, C_out=3, k=3)
    // Output with stride=2: (3, 9)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(4),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=2, C_out=3, k=3)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 3, 3])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // stride=2
    let output = x.conv_transpose1d(&kernel, 2, 0, 0, 1, 1);

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

    eprintln!("ConvTranspose1d (stride=2) Input gradient: {:?}", x.grad());
    eprintln!(
        "ConvTranspose1d (stride=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}

#[test]
fn test_conv_transpose1d_backward_groups() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose1d backward with groups
    // Input: (C_in=4, L=4), Kernel: (C_in=4, C_out/groups=2, k=3)
    // groups=2, so C_out=4
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(4),
                crate::graph::shape::Expr::from(4),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=4, C_out/groups=2, k=3)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![4, 2, 3])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // groups=2
    let output = x.conv_transpose1d(&kernel, 1, 0, 0, 1, 2);

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

    eprintln!("ConvTranspose1d (groups=2) Input gradient: {:?}", x.grad());
    eprintln!(
        "ConvTranspose1d (groups=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}

#[test]
fn test_conv_transpose3d_backward_simple() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose3d backward (simple case)
    // Input: (C_in=2, D=2, H=2, W=2), Kernel: (C_in=2, C_out=3, kD=2, kH=2, kW=2)
    // Output with stride=1: (3, 3, 3, 3)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=2, C_out=3, kD=2, kH=2, kW=2)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 3, 2, 2, 2])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // stride=1
    let output = x.conv_transpose3d(&kernel, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1), 1);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("ConvTranspose3d Input gradient: {:?}", x.grad());
    eprintln!("ConvTranspose3d Kernel gradient: {:?}", kernel.grad());
}

#[test]
fn test_conv_transpose3d_backward_stride() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose3d backward with stride
    // Input: (C_in=2, D=2, H=2, W=2), Kernel: (C_in=2, C_out=3, kD=2, kH=2, kW=2)
    // Output with stride=2: (3, 5, 5, 5)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=2, C_out=3, kD=2, kH=2, kW=2)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![2, 3, 2, 2, 2])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // stride=2
    let output = x.conv_transpose3d(&kernel, (2, 2, 2), (0, 0, 0), (0, 0, 0), (1, 1, 1), 1);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("ConvTranspose3d (stride=2) Input gradient: {:?}", x.grad());
    eprintln!(
        "ConvTranspose3d (stride=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}

#[test]
fn test_conv_transpose3d_backward_groups() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ConvTranspose3d backward with groups
    // Input: (C_in=4, D=2, H=2, W=2), Kernel: (C_in=4, C_out/groups=2, kD=2, kH=2, kW=2)
    // groups=2, so C_out=4
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(4),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
                crate::graph::shape::Expr::from(2),
            ]),
        true, // requires_grad
    );

    // Kernel: (C_in=4, C_out/groups=2, kD=2, kH=2, kW=2)
    let mut kernel_graph = crate::graph::Graph::new();
    let kernel_node = kernel_graph
        .input("kernel")
        .with_dtype(crate::graph::DType::F32)
        .with_shape(vec![4, 2, 2, 2, 2])
        .build();

    let kernel = Tensor::from_graph_node(kernel_node, true);

    // groups=2
    let output = x.conv_transpose3d(&kernel, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1), 2);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0).sum(0).sum(0);

    // backwardを実行
    scalar.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(
        kernel.grad().is_some(),
        "Kernel gradient should be computed"
    );

    eprintln!("ConvTranspose3d (groups=2) Input gradient: {:?}", x.grad());
    eprintln!(
        "ConvTranspose3d (groups=2) Kernel gradient: {:?}",
        kernel.grad()
    );
}
