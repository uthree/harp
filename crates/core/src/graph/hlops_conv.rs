//! 畳み込み演算
//!
//! このモジュールは統一されたconv/conv_transpose APIを提供します。

use crate::graph::GraphNode;
use crate::graph::conv::{ConvParams, IntoSpatialParams};

impl GraphNode {
    /// 畳み込み演算
    ///
    /// N次元畳み込みを統一APIで実行します。次元数は入力形状から自動判定されます。
    /// groupsはカーネル形状から自動計算されます（groups = C_in / kernel.shape[1]）。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, k1, k2, ...)
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `padding`: パディング（現在は0のみサポート）
    ///
    /// # 入出力形状
    /// - 入力: (C_in, L1, L2, ...)
    /// - カーネル: (C_out, C_in/groups, k1, k2, ...)
    /// - 出力: (C_out, L1', L2', ...)
    ///
    /// # 例
    /// ```no_run
    /// use harp_core::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![3, 32, 32]);
    /// let kernel = graph.input("kernel", DType::F32, vec![16, 3, 3, 3]);
    ///
    /// // 2D conv: (3, 32, 32) conv (16, 3, 3, 3) -> (16, 30, 30)
    /// let output = x.conv(kernel, (1, 1), (1, 1), (0, 0));
    /// ```
    pub fn conv<S: IntoSpatialParams>(
        self,
        kernel: GraphNode,
        stride: S,
        dilation: S,
        padding: S,
    ) -> GraphNode {
        let stride_vec = stride.into_vec();
        let dilation_vec = dilation.into_vec();
        let padding_vec = padding.into_vec();

        let spatial_dims = self.view.ndim() - 1; // チャネル次元を除く

        // パラメータの次元数を検証
        assert_eq!(
            stride_vec.len(),
            spatial_dims,
            "stride must have {} elements for {}D conv",
            spatial_dims,
            spatial_dims
        );
        assert_eq!(
            dilation_vec.len(),
            spatial_dims,
            "dilation must have {} elements for {}D conv",
            spatial_dims,
            spatial_dims
        );
        assert_eq!(
            padding_vec.len(),
            spatial_dims,
            "padding must have {} elements for {}D conv",
            spatial_dims,
            spatial_dims
        );

        // 現在はpadding=0のみサポート
        for &p in &padding_vec {
            assert_eq!(
                p, 0,
                "padding must be 0 (padding support not yet implemented)"
            );
        }

        // groupsをカーネル形状から自動計算: groups = C_in / kernel.shape[1]
        let c_in = self.view.shape()[0].expect_usize("C_in must be constant");
        let c_in_per_group = kernel.view.shape()[1].expect_usize("C_in/groups must be constant");
        let groups = c_in / c_in_per_group;

        assert!(
            c_in.is_multiple_of(c_in_per_group),
            "C_in ({}) must be divisible by kernel's C_in/groups ({})",
            c_in,
            c_in_per_group
        );

        // ダミーのkernel_sizeを設定（conv_nd内でkernelから取得される）
        let dummy_kernel_size = vec![1; spatial_dims];
        let params = ConvParams::new(dummy_kernel_size, stride_vec, dilation_vec, groups);

        self.conv_nd(kernel, &params)
    }

    /// 転置畳み込み演算（deconvolution / transposed convolution）
    ///
    /// N次元転置畳み込みを統一APIで実行します。主にアップサンプリングに使用されます。
    /// groupsはカーネル形状から自動計算されます。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, k1, k2, ...)
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `padding`: パディング（出力から削られるサイズ）
    ///
    /// # 入出力形状
    /// - 入力: (C_in, L1_in, L2_in, ...)
    /// - カーネル: (C_in, C_out/groups, k1, k2, ...)
    /// - 出力: (C_out, L1_out, L2_out, ...)
    ///   - L_out = (L_in - 1) * s - 2 * p + d * (k - 1) + 1
    ///
    /// # 例
    /// ```no_run
    /// use harp_core::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![16, 8, 8]);
    /// let kernel = graph.input("kernel", DType::F32, vec![16, 3, 3, 3]);
    ///
    /// // 2D conv_transpose: (16, 8, 8) -> (3, 16, 16)
    /// let output = x.conv_transpose(kernel, (2, 2), (1, 1), (0, 0));
    /// ```
    pub fn conv_transpose<S: IntoSpatialParams>(
        self,
        kernel: GraphNode,
        stride: S,
        dilation: S,
        padding: S,
    ) -> GraphNode {
        let stride_vec = stride.into_vec();
        let dilation_vec = dilation.into_vec();
        let padding_vec = padding.into_vec();

        let spatial_dims = self.view.ndim() - 1; // チャネル次元を除く

        // パラメータの次元数を検証
        assert_eq!(
            stride_vec.len(),
            spatial_dims,
            "stride must have {} elements for {}D conv_transpose",
            spatial_dims,
            spatial_dims
        );
        assert_eq!(
            dilation_vec.len(),
            spatial_dims,
            "dilation must have {} elements for {}D conv_transpose",
            spatial_dims,
            spatial_dims
        );
        assert_eq!(
            padding_vec.len(),
            spatial_dims,
            "padding must have {} elements for {}D conv_transpose",
            spatial_dims,
            spatial_dims
        );

        // conv_transpose のカーネル形状: (C_in, C_out/groups, k1, k2, ...)
        // 入力のC_inとカーネルのdim 0が一致することを確認
        let c_in = self.view.shape()[0].expect_usize("C_in must be constant");
        let kernel_c_in = kernel.view.shape()[0].expect_usize("kernel C_in must be constant");

        // groupsを計算: kernel.shape[0]がC_inなので、groups = 1 と仮定
        // （grouped conv_transposeの場合、C_in = groups * c_in_per_group）
        let groups = if c_in == kernel_c_in {
            1
        } else {
            // grouped conv_transposeの場合
            assert!(
                c_in.is_multiple_of(kernel_c_in),
                "C_in ({}) must be divisible by kernel's dim 0 ({})",
                c_in,
                kernel_c_in
            );
            c_in / kernel_c_in
        };

        // output_paddingはデフォルトで0
        let output_padding = vec![0; spatial_dims];

        // ダミーのkernel_sizeを設定（conv_transpose_nd内でkernelから取得される）
        let dummy_kernel_size = vec![1; spatial_dims];
        let params = ConvParams::new(dummy_kernel_size, stride_vec, dilation_vec, groups);

        self.conv_transpose_nd(kernel, &params, padding_vec, output_padding)
    }
}
