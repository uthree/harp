/// 畳み込み操作の統合テスト
///
/// conv1d, conv2d, conv3dがunfold + elementwise-mul + reduceで
/// 正しく動作することを検証します。
#[cfg(test)]
mod tests {
    use harp::graph::shape::Expr;
    use harp::prelude::*;

    #[test]
    fn test_conv1d_basic() {
        let mut graph = Graph::new();

        // Input: [C_in=2, L=5], Kernel: [C_out=3, C_in=2, k=3]
        // Output: [3, 3] (length=(5-3)/1+1=3)
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 5])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 2, 3])
            .build();

        let output = input.conv1d(kernel, 1, 1, 1);

        assert_eq!(output.view.shape(), &[Expr::from(3), Expr::from(3)]);
    }

    #[test]
    fn test_conv1d_stride() {
        let mut graph = Graph::new();

        // Test with stride=2
        // Input: [1, 5], Kernel: [1, 1, 3]
        // Output: [1, 2] (length=(5-3)/2+1=2)
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 5])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 3])
            .build();

        let output = input.conv1d(kernel, 2, 1, 1);

        assert_eq!(output.view.shape(), &[Expr::from(1), Expr::from(2)]);
    }

    #[test]
    fn test_conv1d_dilation() {
        let mut graph = Graph::new();

        // Test with dilation=2
        // Input: [1, 6], Kernel: [1, 1, 3]
        // Effective kernel size = (3-1)*2+1 = 5
        // Output: [1, 2] (length=(6-5)/1+1=2)
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 6])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 3])
            .build();

        let output = input.conv1d(kernel, 1, 2, 1);

        assert_eq!(output.view.shape(), &[Expr::from(1), Expr::from(2)]);
    }

    #[test]
    fn test_conv1d_groups() {
        let mut graph = Graph::new();

        // Test with groups=2 (group convolution)
        // Input: [2, 3], Kernel: [2, 1, 2]
        // Output: [2, 2]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 1, 2])
            .build();

        let output = input.conv1d(kernel, 1, 1, 2);

        assert_eq!(output.view.shape(), &[Expr::from(2), Expr::from(2)]);
    }

    #[test]
    fn test_conv1d_depthwise() {
        let mut graph = Graph::new();

        // Depthwise convolution: groups = in_channels = out_channels
        // Input: [2, 4], Kernel: [2, 1, 2]
        // Output: [2, 3]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 4])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 1, 2])
            .build();

        let output = input.conv1d(kernel, 1, 1, 2);

        assert_eq!(output.view.shape(), &[Expr::from(2), Expr::from(3)]);
    }

    #[test]
    fn test_conv1d_multi_channel() {
        let mut graph = Graph::new();

        // Multiple input and output channels
        // Input: [3, 3], Kernel: [2, 3, 3]
        // Output: [2, 1] (length=(3-3)/1+1=1)
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 3])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3, 3])
            .build();

        let output = input.conv1d(kernel, 1, 1, 1);

        assert_eq!(output.view.shape(), &[Expr::from(2), Expr::from(1)]);
    }

    #[test]
    fn test_conv2d_basic() {
        let mut graph = Graph::new();

        // Input: [C_in=1, H=4, W=4], Kernel: [C_out=1, C_in=1, kH=2, kW=2]
        // Output: [1, 3, 3]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 4, 4])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 2, 2])
            .build();

        let output = input.conv2d(kernel, (1, 1), (1, 1), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(1), Expr::from(3), Expr::from(3)]
        );
    }

    #[test]
    fn test_conv2d_stride() {
        let mut graph = Graph::new();

        // Test with stride=(2, 2)
        // Input: [1, 4, 4], Kernel: [1, 1, 2, 2]
        // Output: [1, 2, 2] (stride=2)
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 4, 4])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 2, 2])
            .build();

        let output = input.conv2d(kernel, (2, 2), (1, 1), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(1), Expr::from(2), Expr::from(2)]
        );
    }

    #[test]
    fn test_conv2d_dilation() {
        let mut graph = Graph::new();

        // Test with dilation=(2, 2)
        // Input: [1, 5, 5], Kernel: [1, 1, 2, 2]
        // Effective kernel size = (2-1)*2+1 = 3 for both dimensions
        // Output: [1, 3, 3]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 5, 5])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 2, 2])
            .build();

        let output = input.conv2d(kernel, (1, 1), (2, 2), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(1), Expr::from(3), Expr::from(3)]
        );
    }

    #[test]
    fn test_conv2d_groups() {
        let mut graph = Graph::new();

        // Test with groups=2
        // Input: [2, 4, 4], Kernel: [2, 1, 2, 2]
        // Output: [2, 3, 3]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 4, 4])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 1, 2, 2])
            .build();

        let output = input.conv2d(kernel, (1, 1), (1, 1), 2);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(2), Expr::from(3), Expr::from(3)]
        );
    }

    #[test]
    fn test_conv2d_depthwise() {
        let mut graph = Graph::new();

        // Depthwise 2D convolution
        // Input: [2, 3, 3], Kernel: [2, 1, 2, 2]
        // Output: [2, 2, 2]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3, 3])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 1, 2, 2])
            .build();

        let output = input.conv2d(kernel, (1, 1), (1, 1), 2);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(2), Expr::from(2), Expr::from(2)]
        );
    }

    #[test]
    fn test_conv2d_multi_channel() {
        let mut graph = Graph::new();

        // Multiple input and output channels
        // Input: [3, 5, 5], Kernel: [16, 3, 3, 3]
        // Output: [16, 3, 3]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 5, 5])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![16, 3, 3, 3])
            .build();

        let output = input.conv2d(kernel, (1, 1), (1, 1), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(16), Expr::from(3), Expr::from(3)]
        );
    }

    #[test]
    fn test_conv3d_basic() {
        let mut graph = Graph::new();

        // Input: [C_in=1, D=3, H=3, W=3], Kernel: [C_out=1, C_in=1, kD=2, kH=2, kW=2]
        // Output: [1, 2, 2, 2]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 3, 3, 3])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 2, 2, 2])
            .build();

        let output = input.conv3d(kernel, (1, 1, 1), (1, 1, 1), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(1), Expr::from(2), Expr::from(2), Expr::from(2)]
        );
    }

    #[test]
    fn test_conv3d_stride() {
        let mut graph = Graph::new();

        // Test with stride=(2, 2, 2)
        // Input: [1, 4, 4, 4], Kernel: [1, 1, 2, 2, 2]
        // Output: [1, 2, 2, 2] (stride=2)
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 4, 4, 4])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 2, 2, 2])
            .build();

        let output = input.conv3d(kernel, (2, 2, 2), (1, 1, 1), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(1), Expr::from(2), Expr::from(2), Expr::from(2)]
        );
    }

    #[test]
    fn test_conv3d_dilation() {
        let mut graph = Graph::new();

        // Test with dilation=(2, 2, 2)
        // Input: [1, 5, 5, 5], Kernel: [1, 1, 2, 2, 2]
        // Effective kernel size = (2-1)*2+1 = 3 for all dimensions
        // Output: [1, 3, 3, 3]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 5, 5, 5])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![1, 1, 2, 2, 2])
            .build();

        let output = input.conv3d(kernel, (1, 1, 1), (2, 2, 2), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(1), Expr::from(3), Expr::from(3), Expr::from(3)]
        );
    }

    #[test]
    fn test_conv3d_groups() {
        let mut graph = Graph::new();

        // Test with groups=2
        // Input: [2, 3, 3, 3], Kernel: [2, 1, 2, 2, 2]
        // Output: [2, 2, 2, 2]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 3, 3, 3])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![2, 1, 2, 2, 2])
            .build();

        let output = input.conv3d(kernel, (1, 1, 1), (1, 1, 1), 2);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(2), Expr::from(2), Expr::from(2), Expr::from(2)]
        );
    }

    #[test]
    fn test_conv3d_multi_channel() {
        let mut graph = Graph::new();

        // Multiple input and output channels
        // Input: [3, 5, 5, 5], Kernel: [8, 3, 3, 3, 3]
        // Output: [8, 3, 3, 3]
        let input = graph
            .input("input")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 5, 5, 5])
            .build();

        let kernel = graph
            .input("kernel")
            .with_dtype(DType::F32)
            .with_shape(vec![8, 3, 3, 3, 3])
            .build();

        let output = input.conv3d(kernel, (1, 1, 1), (1, 1, 1), 1);

        assert_eq!(
            output.view.shape(),
            &[Expr::from(8), Expr::from(3), Expr::from(3), Expr::from(3)]
        );
    }
}
