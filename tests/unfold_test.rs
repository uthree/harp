/// unfold操作の結合テスト
///
/// unfoldがView操作として正しく動作することを検証します。
#[cfg(test)]
mod tests {
    use harp::prelude::*;

    #[test]
    fn test_unfold_1d_basic() {
        let mut graph = Graph::new();

        // 1D入力: (6,) -> unfold(3, 1, 1, 1) -> (3, 4)
        let x = graph.input("x", DType::F32, vec![6]);

        let unfolded = x.unfold(3, 1, 1, 1);

        // shapeが正しいことを確認
        use harp::graph::shape::Expr;
        assert_eq!(unfolded.view.shape(), &[Expr::from(3), Expr::from(4)]);
    }

    #[test]
    fn test_unfold_1d_stride() {
        let mut graph = Graph::new();

        // 1D入力: (10,) -> unfold(3, 2, 1, 1) -> (3, 4)
        // L' = (10 - 3) / 2 + 1 = 4
        let x = graph.input("x", DType::F32, vec![10]);

        let unfolded = x.unfold(3, 2, 1, 1);

        use harp::graph::shape::Expr;
        assert_eq!(unfolded.view.shape(), &[Expr::from(3), Expr::from(4)]);
    }

    #[test]
    fn test_unfold_1d_dilation() {
        let mut graph = Graph::new();

        // 1D入力: (10,) -> unfold(3, 1, 2, 1) -> (3, 6)
        // effective_kernel_size = (3-1)*2+1 = 5
        // L' = (10 - 5) / 1 + 1 = 6
        let x = graph.input("x", DType::F32, vec![10]);

        let unfolded = x.unfold(3, 1, 2, 1);

        use harp::graph::shape::Expr;
        assert_eq!(unfolded.view.shape(), &[Expr::from(3), Expr::from(6)]);
    }

    #[test]
    fn test_unfold_2d_basic() {
        let mut graph = Graph::new();

        // 2D入力: (2, 10) -> unfold(4, 2, 1, 1) -> (2, 4, 4)
        // L' = (10 - 4) / 2 + 1 = 4
        let x = graph.input("x", DType::F32, vec![2, 10]);

        let unfolded = x.unfold(4, 2, 1, 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[Expr::from(2), Expr::from(4), Expr::from(4)]
        );
    }

    #[test]
    fn test_unfold_2d_channels() {
        let mut graph = Graph::new();

        // チャネル付き: (3, 6) -> unfold(3, 1, 1, 1) -> (3, 3, 4)
        let x = graph.input("x", DType::F32, vec![3, 6]);

        let unfolded = x.unfold(3, 1, 1, 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[Expr::from(3), Expr::from(3), Expr::from(4)]
        );
    }

    #[test]
    fn test_unfold_2d_dilation() {
        let mut graph = Graph::new();

        // 2D入力: (2, 15) -> unfold(4, 1, 3, 1) -> (2, 4, 6)
        // effective_kernel_size = (4-1)*3+1 = 10
        // L' = (15 - 10) / 1 + 1 = 6
        let x = graph.input("x", DType::F32, vec![2, 15]);

        let unfolded = x.unfold(4, 1, 3, 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[Expr::from(2), Expr::from(4), Expr::from(6)]
        );
    }

    #[test]
    fn test_unfold_1d_groups() {
        let mut graph = Graph::new();

        // グループ畳み込み: (6, 10) -> unfold(3, 1, 1, 2) -> (2, 3, 3, 8)
        let x = graph.input("x", DType::F32, vec![6, 10]);

        let unfolded = x.unfold(3, 1, 1, 2);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[Expr::from(2), Expr::from(3), Expr::from(3), Expr::from(8)]
        );
    }

    #[test]
    fn test_unfold_1d_depthwise() {
        let mut graph = Graph::new();

        // Depthwise: (4, 10) -> unfold(3, 1, 1, 4) -> (4, 1, 3, 8)
        let x = graph.input("x", DType::F32, vec![4, 10]);

        let unfolded = x.unfold(3, 1, 1, 4);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[Expr::from(4), Expr::from(1), Expr::from(3), Expr::from(8)]
        );
    }

    #[test]
    fn test_unfold_then_reduce() {
        let mut graph = Graph::new();

        // unfold後にreduce操作を適用するユースケース
        // (8,) -> unfold(3, 1, 1, 1) -> (3, 6) -> reduce_sum(axis=0) -> (6,)
        let x = graph.input("x", DType::F32, vec![8]);

        let unfolded = x.unfold(3, 1, 1, 1);
        let summed = unfolded.reduce_sum(0);

        use harp::graph::shape::Expr;
        assert_eq!(summed.view.shape(), &[Expr::from(6)]);
    }

    #[test]
    fn test_unfold_kernel_equals_stride() {
        let mut graph = Graph::new();

        // kernel_size == stride の場合、重複なし
        // (12,) -> unfold(4, 4, 1, 1) -> (4, 3)
        // L' = (12 - 4) / 4 + 1 = 3
        let x = graph.input("x", DType::F32, vec![12]);

        let unfolded = x.unfold(4, 4, 1, 1);

        use harp::graph::shape::Expr;
        assert_eq!(unfolded.view.shape(), &[Expr::from(4), Expr::from(3)]);
    }

    #[test]
    fn test_unfold_1d_combined() {
        let mut graph = Graph::new();

        // dilation + groups の組み合わせ
        // (8, 12) -> unfold(3, 1, 2, 4) -> (4, 2, 3, 8)
        // effective_kernel_size = (3-1)*2+1 = 5
        // L' = (12 - 5) / 1 + 1 = 8
        let x = graph.input("x", DType::F32, vec![8, 12]);

        let unfolded = x.unfold(3, 1, 2, 4);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[Expr::from(4), Expr::from(2), Expr::from(3), Expr::from(8)]
        );
    }

    // === unfold 2D tests ===

    #[test]
    fn test_unfold_2d_spatial_basic() {
        let mut graph = Graph::new();

        // 2D入力: (8, 8) -> unfold((3, 3), (1, 1), (1, 1), 1) -> (3, 3, 6, 6)
        let x = graph.input("x", DType::F32, vec![8, 8]);

        let unfolded = x.unfold((3, 3), (1, 1), (1, 1), 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[Expr::from(3), Expr::from(3), Expr::from(6), Expr::from(6)]
        );
    }

    #[test]
    fn test_unfold_2d_with_channels() {
        let mut graph = Graph::new();

        // 3D入力: (3, 32, 32) -> unfold((3, 3), (1, 1), (1, 1), 1) -> (3, 3, 3, 30, 30)
        let x = graph.input("x", DType::F32, vec![3, 32, 32]);

        let unfolded = x.unfold((3, 3), (1, 1), (1, 1), 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(3),
                Expr::from(3),
                Expr::from(3),
                Expr::from(30),
                Expr::from(30)
            ]
        );
    }

    #[test]
    fn test_unfold_2d_stride() {
        let mut graph = Graph::new();

        // stride付き: (3, 16, 16) -> unfold((3, 3), (2, 2), (1, 1), 1) -> (3, 3, 3, 7, 7)
        // H' = (16 - 3) / 2 + 1 = 7
        let x = graph.input("x", DType::F32, vec![3, 16, 16]);

        let unfolded = x.unfold((3, 3), (2, 2), (1, 1), 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(3),
                Expr::from(3),
                Expr::from(3),
                Expr::from(7),
                Expr::from(7)
            ]
        );
    }

    #[test]
    fn test_unfold_2d_spatial_dilation() {
        let mut graph = Graph::new();

        // dilation付き: (2, 16, 16) -> unfold((3, 3), (1, 1), (2, 2), 1) -> (2, 3, 3, 12, 12)
        // effective_kernel = (3-1)*2+1 = 5
        // H' = (16 - 5) / 1 + 1 = 12
        let x = graph.input("x", DType::F32, vec![2, 16, 16]);

        let unfolded = x.unfold((3, 3), (1, 1), (2, 2), 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(2),
                Expr::from(3),
                Expr::from(3),
                Expr::from(12),
                Expr::from(12)
            ]
        );
    }

    #[test]
    fn test_unfold_2d_groups() {
        let mut graph = Graph::new();

        // groups=2: (6, 16, 16) -> unfold((3, 3), (1, 1), (1, 1), 2) -> (2, 3, 3, 3, 14, 14)
        let x = graph.input("x", DType::F32, vec![6, 16, 16]);

        let unfolded = x.unfold((3, 3), (1, 1), (1, 1), 2);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(2),
                Expr::from(3),
                Expr::from(3),
                Expr::from(3),
                Expr::from(14),
                Expr::from(14)
            ]
        );
    }

    #[test]
    fn test_unfold_2d_depthwise() {
        let mut graph = Graph::new();

        // depthwise: (4, 16, 16) -> unfold((3, 3), (1, 1), (1, 1), 4) -> (4, 1, 3, 3, 14, 14)
        let x = graph.input("x", DType::F32, vec![4, 16, 16]);

        let unfolded = x.unfold((3, 3), (1, 1), (1, 1), 4);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(4),
                Expr::from(1),
                Expr::from(3),
                Expr::from(3),
                Expr::from(14),
                Expr::from(14)
            ]
        );
    }

    // === unfold 3D tests ===

    #[test]
    fn test_unfold_3d_basic() {
        let mut graph = Graph::new();

        // 3D入力: (8, 8, 8) -> unfold((3, 3, 3), (1, 1, 1), (1, 1, 1), 1) -> (3, 3, 3, 6, 6, 6)
        let x = graph.input("x", DType::F32, vec![8, 8, 8]);

        let unfolded = x.unfold((3, 3, 3), (1, 1, 1), (1, 1, 1), 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(3),
                Expr::from(3),
                Expr::from(3),
                Expr::from(6),
                Expr::from(6),
                Expr::from(6)
            ]
        );
    }

    #[test]
    fn test_unfold_3d_with_channels() {
        let mut graph = Graph::new();

        // 4D入力: (2, 16, 16, 16) -> unfold((3, 3, 3), (1, 1, 1), (1, 1, 1), 1) -> (2, 3, 3, 3, 14, 14, 14)
        let x = graph.input("x", DType::F32, vec![2, 16, 16, 16]);

        let unfolded = x.unfold((3, 3, 3), (1, 1, 1), (1, 1, 1), 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(2),
                Expr::from(3),
                Expr::from(3),
                Expr::from(3),
                Expr::from(14),
                Expr::from(14),
                Expr::from(14)
            ]
        );
    }

    #[test]
    fn test_unfold_3d_stride() {
        let mut graph = Graph::new();

        // stride付き: (2, 16, 16, 16) -> unfold((3, 3, 3), (2, 2, 2), (1, 1, 1), 1) -> (2, 3, 3, 3, 7, 7, 7)
        let x = graph.input("x", DType::F32, vec![2, 16, 16, 16]);

        let unfolded = x.unfold((3, 3, 3), (2, 2, 2), (1, 1, 1), 1);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(2),
                Expr::from(3),
                Expr::from(3),
                Expr::from(3),
                Expr::from(7),
                Expr::from(7),
                Expr::from(7)
            ]
        );
    }

    #[test]
    fn test_unfold_3d_groups() {
        let mut graph = Graph::new();

        // groups=2: (4, 12, 12, 12) -> unfold((3, 3, 3), (1, 1, 1), (1, 1, 1), 2) -> (2, 2, 3, 3, 3, 10, 10, 10)
        let x = graph.input("x", DType::F32, vec![4, 12, 12, 12]);

        let unfolded = x.unfold((3, 3, 3), (1, 1, 1), (1, 1, 1), 2);

        use harp::graph::shape::Expr;
        assert_eq!(
            unfolded.view.shape(),
            &[
                Expr::from(2),
                Expr::from(2),
                Expr::from(3),
                Expr::from(3),
                Expr::from(3),
                Expr::from(10),
                Expr::from(10),
                Expr::from(10)
            ]
        );
    }
}
