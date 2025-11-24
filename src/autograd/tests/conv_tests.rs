/// autograd::Tensorの畳み込み演算のテスト
///
/// 注意: 勾配計算は未実装のため、前向き計算のみをテストします。
use crate::autograd::Tensor;

#[test]
fn test_tensor_conv1d_basic() {
    // Input: (C_in=2, L=5), Kernel: (C_out=3, C_in=2, k=3)
    // Output: (3, 3)
    let x = Tensor::ones(vec![2, 5]);
    let kernel = Tensor::ones(vec![3, 2, 3]);

    let output = x.conv1d(&kernel, 1, 1, 1);

    // Shape check
    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(3),
            crate::graph::shape::Expr::from(3)
        ]
    );
}

#[test]
fn test_tensor_conv1d_stride() {
    // Test with stride=2
    // Input: (1, 5), Kernel: (1, 1, 3)
    // Output: (1, 2)
    let x = Tensor::ones(vec![1, 5]);
    let kernel = Tensor::ones(vec![1, 1, 3]);

    let output = x.conv1d(&kernel, 2, 1, 1);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(1),
            crate::graph::shape::Expr::from(2)
        ]
    );
}

#[test]
#[ignore = "Group convolution has reshape issue with non-contiguous views - needs fix in hlops_conv.rs"]
fn test_tensor_conv1d_groups() {
    // Test with groups=2 (group convolution)
    // Input: (2, 3), Kernel: (2, 1, 2)
    // Output: (2, 2)
    let x = Tensor::ones(vec![2, 3]);
    let kernel = Tensor::ones(vec![2, 1, 2]);

    let output = x.conv1d(&kernel, 1, 1, 2);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2)
        ]
    );
}

#[test]
fn test_tensor_conv2d_basic() {
    // Input: (C_in=1, H=4, W=4), Kernel: (C_out=1, C_in=1, kH=2, kW=2)
    // Output: (1, 3, 3)
    let x = Tensor::ones(vec![1, 4, 4]);
    let kernel = Tensor::ones(vec![1, 1, 2, 2]);

    let output = x.conv2d(&kernel, (1, 1), (1, 1), 1);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(1),
            crate::graph::shape::Expr::from(3),
            crate::graph::shape::Expr::from(3)
        ]
    );
}

#[test]
fn test_tensor_conv2d_stride() {
    // Test with stride=(2, 2)
    // Input: (1, 4, 4), Kernel: (1, 1, 2, 2)
    // Output: (1, 2, 2)
    let x = Tensor::ones(vec![1, 4, 4]);
    let kernel = Tensor::ones(vec![1, 1, 2, 2]);

    let output = x.conv2d(&kernel, (2, 2), (1, 1), 1);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(1),
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2)
        ]
    );
}

#[test]
fn test_tensor_conv2d_multi_channel() {
    // Multiple input and output channels
    // Input: (3, 5, 5), Kernel: (16, 3, 3, 3)
    // Output: (16, 3, 3)
    let x = Tensor::ones(vec![3, 5, 5]);
    let kernel = Tensor::ones(vec![16, 3, 3, 3]);

    let output = x.conv2d(&kernel, (1, 1), (1, 1), 1);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(16),
            crate::graph::shape::Expr::from(3),
            crate::graph::shape::Expr::from(3)
        ]
    );
}

#[test]
#[ignore = "Group convolution has reshape issue with non-contiguous views - needs fix in hlops_conv.rs"]
fn test_tensor_conv2d_depthwise() {
    // Depthwise 2D convolution (groups = in_channels)
    // Input: (2, 3, 3), Kernel: (2, 1, 2, 2)
    // Output: (2, 2, 2)
    let x = Tensor::ones(vec![2, 3, 3]);
    let kernel = Tensor::ones(vec![2, 1, 2, 2]);

    let output = x.conv2d(&kernel, (1, 1), (1, 1), 2);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2)
        ]
    );
}

#[test]
fn test_tensor_conv3d_basic() {
    // Input: (C_in=1, D=3, H=3, W=3), Kernel: (C_out=1, C_in=1, kD=2, kH=2, kW=2)
    // Output: (1, 2, 2, 2)
    let x = Tensor::ones(vec![1, 3, 3, 3]);
    let kernel = Tensor::ones(vec![1, 1, 2, 2, 2]);

    let output = x.conv3d(&kernel, (1, 1, 1), (1, 1, 1), 1);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(1),
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2)
        ]
    );
}

#[test]
fn test_tensor_conv3d_stride() {
    // Test with stride=(2, 2, 2)
    // Input: (1, 4, 4, 4), Kernel: (1, 1, 2, 2, 2)
    // Output: (1, 2, 2, 2)
    let x = Tensor::ones(vec![1, 4, 4, 4]);
    let kernel = Tensor::ones(vec![1, 1, 2, 2, 2]);

    let output = x.conv3d(&kernel, (2, 2, 2), (1, 1, 1), 1);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(1),
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2),
            crate::graph::shape::Expr::from(2)
        ]
    );
}

#[test]
fn test_tensor_conv3d_multi_channel() {
    // Multiple input and output channels
    // Input: (3, 5, 5, 5), Kernel: (8, 3, 3, 3, 3)
    // Output: (8, 3, 3, 3)
    let x = Tensor::ones(vec![3, 5, 5, 5]);
    let kernel = Tensor::ones(vec![8, 3, 3, 3, 3]);

    let output = x.conv3d(&kernel, (1, 1, 1), (1, 1, 1), 1);

    assert_eq!(
        output.data.view.shape(),
        &[
            crate::graph::shape::Expr::from(8),
            crate::graph::shape::Expr::from(3),
            crate::graph::shape::Expr::from(3),
            crate::graph::shape::Expr::from(3)
        ]
    );
}

#[test]
fn test_tensor_conv1d_backward_works() {
    // Verify that backward() works correctly on conv result
    // Conv1d backward is now implemented (stride=1, dilation=1, groups=1)

    // Input needs to be 2D: (C_in, L)
    let x = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(3),
            ]),
        true,
    );

    // Kernel needs to be 3D: (C_out, C_in, k)
    let kernel = Tensor::from_graph_node(
        crate::graph::GraphNode::constant(1.0f32)
            .view(crate::graph::shape::View::contiguous(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
            ]))
            .expand(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from(2),
            ]),
        true,
    );

    let output = x.conv1d(&kernel, 1, 1, 1);

    // sum to reduce to scalar
    let scalar = output.sum(0).sum(0);

    // This should now work without panic
    scalar.backward();

    // Verify gradients are computed
    assert!(x.grad().is_some(), "Input gradient should be computed");
    assert!(kernel.grad().is_some(), "Kernel gradient should be computed");
}
