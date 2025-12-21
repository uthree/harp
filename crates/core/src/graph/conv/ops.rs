//! N次元畳み込み操作の共通実装
//!
//! conv_nd/conv_transpose_ndを提供します。

use super::ConvParams;
use crate::graph::shape::{Expr, View};
use crate::graph::{GraphNode, GraphOp};

impl GraphNode {
    /// N次元畳み込み（内部API）
    ///
    /// unfold、elementwise乗算、reduceを組み合わせてN次元畳み込みを実装します。
    /// 1D/2D/3D畳み込みの共通実装です。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, k1, k2, ...)
    /// - `params`: 畳み込みパラメータ（stride, dilation, groups - kernel_sizeはkernelから取得）
    ///
    /// # 入出力形状
    /// - 入力: (C_in, L1, L2, ...)
    /// - カーネル: (C_out, C_in/groups, k1, k2, ...)
    /// - 出力: (C_out, L1', L2', ...)
    ///
    /// 通常は`conv`メソッドを使用してください：
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
    pub fn conv_nd(self, kernel: GraphNode, params: &ConvParams) -> GraphNode {
        let spatial_dims = params.ndim();

        // 入力の検証
        assert_eq!(
            self.view.ndim(),
            spatial_dims + 1,
            "conv_nd: input must be {}D (C_in, {})",
            spatial_dims + 1,
            (0..spatial_dims)
                .map(|i| format!("L{}", i))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert_eq!(
            kernel.view.ndim(),
            spatial_dims + 2,
            "conv_nd: kernel must be {}D (C_out, C_in/groups, {})",
            spatial_dims + 2,
            (0..spatial_dims)
                .map(|i| format!("k{}", i))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // カーネルサイズをkernelから取得
        let kernel_sizes: Vec<usize> = (0..spatial_dims)
            .map(|i| match &kernel.view.shape()[2 + i] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            })
            .collect();

        // paramsにカーネルサイズを設定した新しいConvParamsを作成
        let conv_params = ConvParams::new(
            kernel_sizes,
            params.stride.clone(),
            params.dilation.clone(),
            params.groups,
        );

        if params.groups == 1 {
            self.conv_nd_impl(kernel, &conv_params)
        } else {
            self.conv_nd_grouped(kernel, &conv_params)
        }
    }

    /// 通常畳み込み（groups=1）の実装
    fn conv_nd_impl(self, kernel: GraphNode, params: &ConvParams) -> GraphNode {
        let spatial_dims = params.ndim();

        // unfold: (C_in, L1, L2, ...) -> (C_in, k1, k2, ..., L1', L2', ...)
        let unfolded = self.unfold_nd(params);

        // unsqueeze(0): (C_in, k1, ..., L1', ...) -> (1, C_in, k1, ..., L1', ...)
        let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(0));

        // kernel: (C_out, C_in, k1, ...) -> unsqueeze multiple times -> (C_out, C_in, k1, ..., 1, 1, ...)
        let mut kernel_view = kernel.view.clone();
        for _ in 0..spatial_dims {
            let ndim = kernel_view.ndim();
            kernel_view = kernel_view.unsqueeze(ndim);
        }
        let kernel_expanded = kernel.view(kernel_view);

        // 共通シェイプを構築: (C_out, C_in, k1, ..., L1', ...)
        let c_out = kernel.view.shape()[0].clone();
        let c_in = unfolded.view.shape()[0].clone();

        let mut common_shape = vec![c_out.clone(), c_in.clone()];
        // カーネルサイズ
        for i in 0..spatial_dims {
            common_shape.push(unfolded.view.shape()[1 + i].clone());
        }
        // 出力サイズ
        for i in 0..spatial_dims {
            common_shape.push(unfolded.view.shape()[1 + spatial_dims + i].clone());
        }

        let unfolded_broadcasted = unfolded_expanded.broadcast_to(common_shape.clone());
        let kernel_broadcasted = kernel_expanded.broadcast_to(common_shape);

        // 乗算
        let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

        // reduce_sum: C_in (axis=1)と各カーネル次元(axis=2,3,...)をreduce
        let mut result = mul_result;
        for _ in 0..(spatial_dims + 1) {
            result = result.reduce_sum(1);
        }

        result
    }

    /// グループ畳み込み（groups>1）の実装
    fn conv_nd_grouped(self, kernel: GraphNode, params: &ConvParams) -> GraphNode {
        let spatial_dims = params.ndim();
        let groups = params.groups;

        let c_in_per_group =
            match (self.view.shape()[0].clone() / Expr::from(groups as i64)).simplify() {
                Expr::Const(c) => c as usize,
                _ => panic!("C_in/groups must be constant"),
            };

        // unfold with groups: (C_in, L1, ...) -> (groups, C_in/groups, k1, ..., L1', ...)
        let unfolded = self.unfold_nd(params);

        // unsqueeze(1): (groups, C_in/g, k1, ..., L1', ...) -> (groups, 1, C_in/g, k1, ..., L1', ...)
        let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(1));

        // kernel: (C_out, C_in/g, k1, ...) を (groups, C_out/g, C_in/g, k1, ...) にreshape
        let c_out = kernel.view.shape()[0].clone();
        let c_out_per_group = (c_out.clone() / Expr::from(groups as i64)).simplify();

        // contiguous化
        let kernel_contiguous_view = View::contiguous(kernel.view.shape().to_vec());
        let kernel_contiguous = GraphNode::new(
            kernel.dtype.clone(),
            GraphOp::Contiguous {},
            vec![kernel.clone()],
            kernel_contiguous_view,
        );

        // reshape: (C_out, C_in/g, k1, ...) -> (groups, C_out/g, C_in/g, k1, ...)
        let mut reshape_shape = vec![
            Expr::from(groups as i64),
            c_out_per_group.clone(),
            Expr::from(c_in_per_group as i64),
        ];
        for &ks in &params.kernel_size {
            reshape_shape.push(Expr::from(ks as i64));
        }
        let kernel_reshaped = kernel_contiguous.reshape(reshape_shape);

        // unsqueeze multiple times at the end for output spatial dimensions
        let mut kernel_view = kernel_reshaped.view.clone();
        for _ in 0..spatial_dims {
            let ndim = kernel_view.ndim();
            kernel_view = kernel_view.unsqueeze(ndim);
        }
        let kernel_expanded = kernel_reshaped.view(kernel_view);

        // 共通シェイプ: (groups, C_out/g, C_in/g, k1, ..., L1', ...)
        let mut common_shape = vec![
            Expr::from(groups as i64),
            c_out_per_group.clone(),
            Expr::from(c_in_per_group as i64),
        ];
        for &ks in &params.kernel_size {
            common_shape.push(Expr::from(ks as i64));
        }
        // 出力サイズ (unfoldedのshape: groups, C_in/g, k1, ..., L1', ...)
        for i in 0..spatial_dims {
            common_shape.push(unfolded.view.shape()[2 + spatial_dims + i].clone());
        }

        let unfolded_broadcasted = unfolded_expanded.broadcast_to(common_shape.clone());
        let kernel_broadcasted = kernel_expanded.broadcast_to(common_shape);

        // 乗算
        let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

        // reduce_sum: C_in/g (axis=2)と各カーネル次元をreduce
        let mut result = mul_result;
        for _ in 0..(spatial_dims + 1) {
            result = result.reduce_sum(2);
        }

        // reshape: (groups, C_out/g, L1', ...) -> (C_out, L1', ...)
        let mut final_shape = vec![c_out];
        for i in 0..spatial_dims {
            final_shape.push(result.view.shape()[2 + i].clone());
        }

        result.reshape(final_shape)
    }

    /// N次元転置畳み込み（内部API）
    ///
    /// 畳み込みの逆操作を行います。主にアップサンプリングに使用されます。
    /// 1D/2D/3D転置畳み込みの共通実装です。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_in, C_out/groups, k1, k2, ...)
    /// - `params`: 畳み込みパラメータ（stride, dilation, groups - kernel_sizeはkernelから取得）
    /// - `padding`: パディング - 出力から削られるサイズ
    /// - `output_padding`: 出力パディング - 出力に追加されるサイズ
    ///
    /// # 入出力形状
    /// - 入力: (C_in, L1_in, L2_in, ...)
    /// - カーネル: (C_in, C_out/groups, k1, k2, ...)
    /// - 出力: (C_out, L1_out, L2_out, ...)
    ///   - L_out = (L_in - 1) * s - 2 * p + d * (k - 1) + op + 1
    ///
    /// 通常は`conv_transpose`メソッドを使用してください：
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
    pub fn conv_transpose_nd(
        self,
        kernel: GraphNode,
        params: &ConvParams,
        padding: Vec<usize>,
        output_padding: Vec<usize>,
    ) -> GraphNode {
        let spatial_dims = params.ndim();

        // 入力の検証
        assert_eq!(
            self.view.ndim(),
            spatial_dims + 1,
            "conv_transpose_nd: input must be {}D (C_in, {})",
            spatial_dims + 1,
            (0..spatial_dims)
                .map(|i| format!("L{}_in", i))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert_eq!(
            kernel.view.ndim(),
            spatial_dims + 2,
            "conv_transpose_nd: kernel must be {}D (C_in, C_out/groups, {})",
            spatial_dims + 2,
            (0..spatial_dims)
                .map(|i| format!("k{}", i))
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert_eq!(
            padding.len(),
            spatial_dims,
            "padding must have {} elements",
            spatial_dims
        );
        assert_eq!(
            output_padding.len(),
            spatial_dims,
            "output_padding must have {} elements",
            spatial_dims
        );

        // 形状情報を取得
        let input_shape = self.view.shape();
        let kernel_shape = kernel.view.shape();

        let c_in = input_shape[0].expect_usize("C_in must be constant");
        let input_sizes: Vec<usize> = (0..spatial_dims)
            .map(|i| input_shape[1 + i].expect_usize(&format!("L{}_in must be constant", i)))
            .collect();

        let kernel_c_in = kernel_shape[0].expect_usize("kernel C_in must be constant");
        let c_out_per_group = kernel_shape[1].expect_usize("C_out/groups must be constant");
        let kernel_sizes: Vec<usize> = (0..spatial_dims)
            .map(|i| kernel_shape[2 + i].expect_usize(&format!("k{} must be constant", i)))
            .collect();

        assert_eq!(
            c_in, kernel_c_in,
            "Input channels must match kernel input channels"
        );

        let c_out = c_out_per_group * params.groups;

        // 出力サイズを計算: L_out = (L_in - 1) * s - 2 * p + d * (k - 1) + op + 1
        let output_sizes: Vec<usize> = (0..spatial_dims)
            .map(|i| {
                (input_sizes[i] - 1) * params.stride[i]
                    + params.dilation[i] * (kernel_sizes[i] - 1)
                    + output_padding[i]
                    + 1
                    - 2 * padding[i]
            })
            .collect();

        // paramsにカーネルサイズを設定した新しいConvParamsを作成
        let conv_params = ConvParams::new(
            kernel_sizes.clone(),
            params.stride.clone(),
            params.dilation.clone(),
            params.groups,
        );

        let result = if params.groups == 1 {
            self.conv_transpose_nd_impl(
                kernel,
                &conv_params,
                c_in,
                c_out,
                &input_sizes,
                &output_sizes,
            )
        } else {
            self.conv_transpose_nd_grouped(
                kernel,
                &conv_params,
                c_in,
                c_out,
                c_out_per_group,
                &input_sizes,
                &output_sizes,
            )
        };

        // paddingがある場合はスライスで削る
        let has_padding = padding.iter().any(|&p| p > 0);
        if has_padding {
            let mut slice_ranges = vec![(0, c_out)];
            for i in 0..spatial_dims {
                let start = padding[i];
                let end = output_sizes[i] + 2 * padding[i] - padding[i];
                slice_ranges.push((start, end));
            }
            result.slice(slice_ranges)
        } else {
            result
        }
    }

    /// 通常転置畳み込み（groups=1）の実装
    fn conv_transpose_nd_impl(
        self,
        kernel: GraphNode,
        params: &ConvParams,
        c_in: usize,
        c_out: usize,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> GraphNode {
        let spatial_dims = params.ndim();

        // 入力をunsqueeze: (C_in, L1_in, ...) -> (C_in, 1, 1, ..., L1_in, ...)
        // 1を spatial_dims 回挿入
        let mut input_view = self.view.clone();
        for _ in 0..spatial_dims {
            input_view = input_view.unsqueeze(1);
        }
        let input_expanded = self.view(input_view);

        // カーネルをunsqueeze: (C_in, C_out, k1, ...) -> (C_in, C_out, k1, ..., 1, ...)
        // 末尾に1を spatial_dims 回挿入
        let mut kernel_view = kernel.view.clone();
        for _ in 0..spatial_dims {
            let ndim = kernel_view.ndim();
            kernel_view = kernel_view.unsqueeze(ndim);
        }
        let kernel_expanded = kernel.view(kernel_view);

        // 共通シェイプに展開: (C_in, C_out, k1, ..., L1_in, ...)
        let mut common_shape = vec![Expr::from(c_in as i64), Expr::from(c_out as i64)];
        for &ks in &params.kernel_size {
            common_shape.push(Expr::from(ks as i64));
        }
        for &is in input_sizes {
            common_shape.push(Expr::from(is as i64));
        }

        let input_bc = input_expanded.broadcast_to(common_shape.clone());
        let kernel_bc = kernel_expanded.broadcast_to(common_shape);

        // 乗算: (C_in, C_out, k1, ..., L1_in, ...)
        let multiplied = &input_bc * &kernel_bc;

        // C_in軸でreduce: (C_out, k1, ..., L1_in, ...)
        let reduced = multiplied.reduce_sum(0);

        // reshape for fold: (C_out, k1, ..., L1_in, ...) -> (1, C_out * k1 * ..., L1_in * ...)
        let kernel_product: usize = params.kernel_size.iter().product();
        let input_product: usize = input_sizes.iter().product();

        // contiguous化してからreshape
        let reduced_cont_view = View::contiguous(reduced.view.shape().to_vec());
        let reduced_cont = GraphNode::new(
            reduced.dtype.clone(),
            GraphOp::Contiguous {},
            vec![reduced.clone()],
            reduced_cont_view,
        );
        let reshaped = reduced_cont.reshape(vec![
            Expr::from(1),
            Expr::from((c_out * kernel_product) as i64),
            Expr::from(input_product as i64),
        ]);

        // fold: 出力形状に畳み込む
        let mut fold_output_size = vec![c_out];
        fold_output_size.extend_from_slice(output_sizes);

        reshaped.fold_nd(fold_output_size, &params.with_groups(1))
    }

    /// グループ転置畳み込み（groups>1）の実装
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose_nd_grouped(
        self,
        kernel: GraphNode,
        params: &ConvParams,
        c_in: usize,
        c_out: usize,
        c_out_per_group: usize,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> GraphNode {
        let spatial_dims = params.ndim();
        let groups = params.groups;
        let c_in_per_group = c_in / groups;

        // 入力をreshape: (C_in, L1_in, ...) -> (groups, C_in/groups, L1_in, ...)
        let input_contiguous_view = View::contiguous(self.view.shape().to_vec());
        let input_contiguous = GraphNode::new(
            self.dtype.clone(),
            GraphOp::Contiguous {},
            vec![self.clone()],
            input_contiguous_view,
        );
        let mut input_reshape_shape =
            vec![Expr::from(groups as i64), Expr::from(c_in_per_group as i64)];
        for &is in input_sizes {
            input_reshape_shape.push(Expr::from(is as i64));
        }
        let input_reshaped = input_contiguous.reshape(input_reshape_shape);

        // 入力をunsqueeze: (groups, C_in/groups, L1_in, ...) -> (groups, C_in/groups, 1, 1, ..., L1_in, ...)
        // 1を spatial_dims + 1 回挿入（C_out/groupsとk1, k2, ...用）
        let mut input_view = input_reshaped.view.clone();
        for _ in 0..(spatial_dims + 1) {
            input_view = input_view.unsqueeze(2);
        }
        let input_expanded = input_reshaped.view(input_view);

        // カーネルをreshape: (C_in, C_out/groups, k1, ...) -> (groups, C_in/groups, C_out/groups, k1, ...)
        let kernel_contiguous_view = View::contiguous(kernel.view.shape().to_vec());
        let kernel_contiguous = GraphNode::new(
            kernel.dtype.clone(),
            GraphOp::Contiguous {},
            vec![kernel.clone()],
            kernel_contiguous_view,
        );
        let mut kernel_reshape_shape = vec![
            Expr::from(groups as i64),
            Expr::from(c_in_per_group as i64),
            Expr::from(c_out_per_group as i64),
        ];
        for &ks in &params.kernel_size {
            kernel_reshape_shape.push(Expr::from(ks as i64));
        }
        let kernel_reshaped = kernel_contiguous.reshape(kernel_reshape_shape);

        // カーネルをunsqueeze: (groups, C_in/groups, C_out/groups, k1, ...) -> (groups, C_in/groups, C_out/groups, k1, ..., 1, ...)
        let mut kernel_view = kernel_reshaped.view.clone();
        for _ in 0..spatial_dims {
            let ndim = kernel_view.ndim();
            kernel_view = kernel_view.unsqueeze(ndim);
        }
        let kernel_expanded = kernel_reshaped.view(kernel_view);

        // 共通シェイプに展開: (groups, C_in/groups, C_out/groups, k1, ..., L1_in, ...)
        let mut common_shape = vec![
            Expr::from(groups as i64),
            Expr::from(c_in_per_group as i64),
            Expr::from(c_out_per_group as i64),
        ];
        for &ks in &params.kernel_size {
            common_shape.push(Expr::from(ks as i64));
        }
        for &is in input_sizes {
            common_shape.push(Expr::from(is as i64));
        }

        let input_bc = input_expanded.broadcast_to(common_shape.clone());
        let kernel_bc = kernel_expanded.broadcast_to(common_shape);

        // 乗算: (groups, C_in/groups, C_out/groups, k1, ..., L1_in, ...)
        let multiplied = &input_bc * &kernel_bc;

        // C_in/groups軸でreduce: (groups, C_out/groups, k1, ..., L1_in, ...)
        let reduced = multiplied.reduce_sum(1);

        // fold with groups
        let mut fold_output_size = vec![c_out];
        fold_output_size.extend_from_slice(output_sizes);

        reduced.fold_nd(fold_output_size, params)
    }
}
