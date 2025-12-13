//! 畳み込み演算
//!
//! このモジュールは conv1d, conv2d, conv3d などの畳み込み演算を提供します。

use crate::graph::GraphNode;
use crate::graph::conv::ConvParams;

impl GraphNode {
    /// 1D畳み込み
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, k)
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: (C_in, L)
    /// - カーネル: (C_out, C_in/groups, k)
    /// - 出力: (C_out, L')
    pub fn conv1d(
        self,
        kernel: GraphNode,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> GraphNode {
        // kernel_sizeはconv_nd内でkernelから取得されるため、ダミー値を渡す
        let params = ConvParams::from_1d(1, stride, dilation, groups);
        self.conv_nd(kernel, &params)
    }

    /// 2D畳み込み
    ///
    /// unfold、elementwise乗算、reduceを組み合わせて2D畳み込みを実装します。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, kH, kW)
    /// - `stride`: ストライド (sH, sW)
    /// - `dilation`: 膨張率 (dH, dW)
    /// - `groups`: グループ数（1=通常、C_in=depthwise）
    ///
    /// # 入出力
    /// - 入力: (C_in, H, W)
    /// - カーネル: (C_out, C_in/groups, kH, kW)
    /// - 出力: (C_out, H', W')
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![3, 32, 32]);
    /// let kernel = graph.input("kernel", DType::F32, vec![16, 3, 3, 3]);
    ///
    /// // (3, 32, 32) conv (16, 3, 3, 3) -> (16, 30, 30)
    /// let output = x.conv2d(kernel, (1, 1), (1, 1), 1);
    /// ```
    pub fn conv2d(
        self,
        kernel: GraphNode,
        stride: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> GraphNode {
        // kernel_sizeはconv_nd内でkernelから取得されるため、ダミー値を渡す
        let params = ConvParams::from_2d((1, 1), stride, dilation, groups);
        self.conv_nd(kernel, &params)
    }

    /// 3D畳み込み
    ///
    /// unfold、elementwise乗算、reduceを組み合わせて3D畳み込みを実装します。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, kD, kH, kW)
    /// - `stride`: ストライド (sD, sH, sW)
    /// - `dilation`: 膨張率 (dD, dH, dW)
    /// - `groups`: グループ数（1=通常、C_in=depthwise）
    ///
    /// # 入出力
    /// - 入力: (C_in, D, H, W)
    /// - カーネル: (C_out, C_in/groups, kD, kH, kW)
    /// - 出力: (C_out, D', H', W')
    pub fn conv3d(
        self,
        kernel: GraphNode,
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> GraphNode {
        // kernel_sizeはconv_nd内でkernelから取得されるため、ダミー値を渡す
        let params = ConvParams::from_3d((1, 1, 1), stride, dilation, groups);
        self.conv_nd(kernel, &params)
    }

    /// 1D転置畳み込み（deconvolution / transposed convolution）
    ///
    /// 畳み込みの逆操作を行います。主にアップサンプリングに使用されます。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, k)
    /// - `stride`: ストライド
    /// - `padding`: パディング - 出力から削られるサイズ
    /// - `output_padding`: 出力パディング - 出力に追加されるサイズ
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: (C_in, L_in)
    /// - カーネル: (C_in, C_out/groups, k)
    /// - 出力: (C_out, L_out)
    ///   - L_out = (L_in - 1) * s - 2 * p + d * (k - 1) + op + 1
    pub fn conv_transpose1d(
        self,
        kernel: GraphNode,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> GraphNode {
        // kernel_sizeはconv_transpose_nd内でkernelから取得されるため、ダミー値を渡す
        let params = ConvParams::from_1d(1, stride, dilation, groups);
        self.conv_transpose_nd(kernel, &params, vec![padding], vec![output_padding])
    }

    /// 2D転置畳み込み（deconvolution / transposed convolution）
    ///
    /// 畳み込みの逆操作を行います。主にアップサンプリングに使用されます。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, kH, kW)
    /// - `stride`: ストライド (sH, sW)
    /// - `padding`: パディング (pH, pW) - 出力から削られるサイズ
    /// - `output_padding`: 出力パディング (opH, opW) - 出力に追加されるサイズ
    /// - `dilation`: 膨張率 (dH, dW)
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: (C_in, H_in, W_in)
    /// - カーネル: (C_in, C_out/groups, kH, kW)
    /// - 出力: (C_out, H_out, W_out)
    ///   - H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
    ///   - W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1
    pub fn conv_transpose2d(
        self,
        kernel: GraphNode,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> GraphNode {
        // kernel_sizeはconv_transpose_nd内でkernelから取得されるため、ダミー値を渡す
        let params = ConvParams::from_2d((1, 1), stride, dilation, groups);
        self.conv_transpose_nd(
            kernel,
            &params,
            vec![padding.0, padding.1],
            vec![output_padding.0, output_padding.1],
        )
    }

    /// 3D転置畳み込み（deconvolution / transposed convolution）
    ///
    /// 畳み込みの逆操作を行います。主にアップサンプリングに使用されます。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, kD, kH, kW)
    /// - `stride`: ストライド (sD, sH, sW)
    /// - `padding`: パディング (pD, pH, pW) - 出力から削られるサイズ
    /// - `output_padding`: 出力パディング (opD, opH, opW) - 出力に追加されるサイズ
    /// - `dilation`: 膨張率 (dD, dH, dW)
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: (C_in, D_in, H_in, W_in)
    /// - カーネル: (C_in, C_out/groups, kD, kH, kW)
    /// - 出力: (C_out, D_out, H_out, W_out)
    pub fn conv_transpose3d(
        self,
        kernel: GraphNode,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> GraphNode {
        // kernel_sizeはconv_transpose_nd内でkernelから取得されるため、ダミー値を渡す
        let params = ConvParams::from_3d((1, 1, 1), stride, dilation, groups);
        self.conv_transpose_nd(
            kernel,
            &params,
            vec![padding.0, padding.1, padding.2],
            vec![output_padding.0, output_padding.1, output_padding.2],
        )
    }
}
