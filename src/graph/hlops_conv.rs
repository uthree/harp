//! 畳み込み演算
//!
//! このモジュールは conv1d, conv2d, conv3d などの畳み込み演算を提供します。

use crate::graph::GraphNode;

impl GraphNode {
    pub fn conv1d(
        self,
        kernel: GraphNode,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> GraphNode {
        use crate::graph::shape::Expr;

        assert_eq!(self.view.ndim(), 2, "conv1d: input must be 2D (C_in, L)");
        assert_eq!(
            kernel.view.ndim(),
            3,
            "conv1d: kernel must be 3D (C_out, C_in/groups, k)"
        );

        if groups == 1 {
            // 通常の畳み込み
            // unfold: (C_in, L) -> (C_in, k, L')
            let kernel_size = match &kernel.view.shape()[2] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };
            let unfolded = self.unfold1d(kernel_size, stride, dilation, 1);

            // unfold: (C_in, k, L') -> unsqueeze(0) -> (1, C_in, k, L')
            let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(0));

            // kernel: (C_out, C_in, k) -> unsqueeze -> (C_out, C_in, k, 1)
            let kernel_expanded = kernel.view(kernel.view.clone().unsqueeze(3));

            // expand to common shape: (C_out, C_in, k, L')
            let c_out = kernel.view.shape()[0].clone();
            let c_in = unfolded.view.shape()[0].clone();
            let k = unfolded.view.shape()[1].clone();
            let l_out = unfolded.view.shape()[2].clone();
            let common_shape = vec![c_out.clone(), c_in.clone(), k.clone(), l_out.clone()];

            let unfolded_broadcasted = unfolded_expanded.expand(common_shape.clone());
            let kernel_broadcasted = kernel_expanded.expand(common_shape);

            // mul: (C_out, C_in, k, L')
            let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

            // reduce sum over C_in (axis=1) and k (axis=2)

            mul_result.reduce_sum(1).reduce_sum(1)
        } else {
            // グループ畳み込み
            // unfold: (C_in, L) -> (groups, C_in/groups, k, L')
            let c_in_per_group =
                match (self.view.shape()[0].clone() / Expr::from(groups as isize)).simplify() {
                    Expr::Const(c) => c as usize,
                    _ => panic!("C_in/groups must be constant"),
                };

            let kernel_size = match &kernel.view.shape()[2] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };

            let unfolded = self.unfold1d(kernel_size, stride, dilation, groups);

            // unfold: (groups, C_in/groups, k, L') -> unsqueeze(1) -> (groups, 1, C_in/groups, k, L')
            let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(1));

            // kernel: (C_out, C_in/groups, k) を (groups, C_out/groups, C_in/groups, k) にreshape
            // 注: reshapeの前にcontiguous化が必要（非連続Viewの場合に正しく動作しないため）
            let c_out = kernel.view.shape()[0].clone();
            let c_out_per_group = (c_out.clone() / Expr::from(groups as isize)).simplify();

            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.clone()],
                kernel_contiguous_view,
            );
            let kernel_reshaped = kernel_contiguous.reshape(vec![
                Expr::from(groups as isize),
                c_out_per_group.clone(),
                Expr::from(c_in_per_group as isize),
                Expr::from(kernel_size as isize),
            ]);

            // unsqueeze: (groups, C_out/groups, C_in/groups, k, 1)
            let kernel_expanded = kernel_reshaped.view(kernel_reshaped.view.clone().unsqueeze(4));

            // expand to common shape: (groups, C_out/groups, C_in/groups, k, L')
            let l_out = unfolded.view.shape()[3].clone();
            let common_shape = vec![
                Expr::from(groups as isize),
                c_out_per_group.clone(),
                Expr::from(c_in_per_group as isize),
                Expr::from(kernel_size as isize),
                l_out.clone(),
            ];

            let unfolded_broadcasted = unfolded_expanded.expand(common_shape.clone());
            let kernel_broadcasted = kernel_expanded.expand(common_shape);

            // mul: (groups, C_out/groups, C_in/groups, k, L')
            let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

            // reduce sum over C_in/groups (axis=2) and k (axis=3)
            let reduced = mul_result.reduce_sum(2).reduce_sum(2);

            // reshape: (groups, C_out/groups, L') -> (C_out, L')
            let l_out = reduced.view.shape()[2].clone();

            reduced.reshape(vec![c_out, l_out])
        }
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
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 32, 32])
    ///     .build();
    /// let kernel = graph.input("kernel")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![16, 3, 3, 3])
    ///     .build();
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
        use crate::graph::shape::Expr;

        assert_eq!(self.view.ndim(), 3, "conv2d: input must be 3D (C_in, H, W)");
        assert_eq!(
            kernel.view.ndim(),
            4,
            "conv2d: kernel must be 4D (C_out, C_in/groups, kH, kW)"
        );

        if groups == 1 {
            // 通常の畳み込み
            let kernel_h = match &kernel.view.shape()[2] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };
            let kernel_w = match &kernel.view.shape()[3] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };

            // unfold: (C_in, H, W) -> (C_in, kH, kW, H', W')
            let unfolded = self.unfold2d((kernel_h, kernel_w), stride, dilation, 1);

            // unfold: (C_in, kH, kW, H', W') -> unsqueeze(0) -> (1, C_in, kH, kW, H', W')
            let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(0));

            // kernel: (C_out, C_in, kH, kW) -> unsqueeze -> (C_out, C_in, kH, kW, 1, 1)
            let kernel_tmp = kernel.view(kernel.view.clone().unsqueeze(4));
            let kernel_expanded = kernel_tmp.view(kernel_tmp.view.clone().unsqueeze(5));

            // expand to common shape: (C_out, C_in, kH, kW, H', W')
            let c_out = kernel.view.shape()[0].clone();
            let c_in = unfolded.view.shape()[0].clone();
            let kh = unfolded.view.shape()[1].clone();
            let kw = unfolded.view.shape()[2].clone();
            let h_out = unfolded.view.shape()[3].clone();
            let w_out = unfolded.view.shape()[4].clone();
            let common_shape = vec![
                c_out.clone(),
                c_in.clone(),
                kh.clone(),
                kw.clone(),
                h_out.clone(),
                w_out.clone(),
            ];

            let unfolded_broadcasted = unfolded_expanded.expand(common_shape.clone());
            let kernel_broadcasted = kernel_expanded.expand(common_shape);

            // mul: (C_out, C_in, kH, kW, H', W')
            let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

            // reduce sum over C_in (axis=1), kH (axis=2), kW (axis=3)

            mul_result.reduce_sum(1).reduce_sum(1).reduce_sum(1)
        } else {
            // グループ畳み込み
            let c_in_per_group =
                match (self.view.shape()[0].clone() / Expr::from(groups as isize)).simplify() {
                    Expr::Const(c) => c as usize,
                    _ => panic!("C_in/groups must be constant"),
                };

            let kernel_h = match &kernel.view.shape()[2] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };
            let kernel_w = match &kernel.view.shape()[3] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };

            // unfold: (C_in, H, W) -> (groups, C_in/groups, kH, kW, H', W')
            let unfolded = self.unfold2d((kernel_h, kernel_w), stride, dilation, groups);

            // unfold: (groups, C_in/groups, kH, kW, H', W') -> unsqueeze(1) -> (groups, 1, C_in/groups, kH, kW, H', W')
            let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(1));

            // kernel: (C_out, C_in/groups, kH, kW) を (groups, C_out/groups, C_in/groups, kH, kW) にreshape
            // 注: reshapeの前にcontiguous化が必要（非連続Viewの場合に正しく動作しないため）
            let c_out = kernel.view.shape()[0].clone();
            let c_out_per_group = (c_out.clone() / Expr::from(groups as isize)).simplify();

            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.clone()],
                kernel_contiguous_view,
            );
            let kernel_reshaped = kernel_contiguous.reshape(vec![
                Expr::from(groups as isize),
                c_out_per_group.clone(),
                Expr::from(c_in_per_group as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
            ]);

            // unsqueeze: (groups, C_out/groups, C_in/groups, kH, kW, 1, 1)
            let kernel_tmp = kernel_reshaped.view(kernel_reshaped.view.clone().unsqueeze(5));
            let kernel_expanded = kernel_tmp.view(kernel_tmp.view.clone().unsqueeze(6));

            // expand to common shape: (groups, C_out/groups, C_in/groups, kH, kW, H', W')
            let h_out = unfolded.view.shape()[4].clone();
            let w_out = unfolded.view.shape()[5].clone();
            let common_shape = vec![
                Expr::from(groups as isize),
                c_out_per_group.clone(),
                Expr::from(c_in_per_group as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
                h_out.clone(),
                w_out.clone(),
            ];

            let unfolded_broadcasted = unfolded_expanded.expand(common_shape.clone());
            let kernel_broadcasted = kernel_expanded.expand(common_shape);

            // mul: (groups, C_out/groups, C_in/groups, kH, kW, H', W')
            let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

            // reduce sum over C_in/groups (axis=2), kH (axis=3), kW (axis=4)
            let reduced = mul_result.reduce_sum(2).reduce_sum(2).reduce_sum(2);

            // reshape: (groups, C_out/groups, H', W') -> (C_out, H', W')
            let h_out = reduced.view.shape()[2].clone();
            let w_out = reduced.view.shape()[3].clone();

            reduced.reshape(vec![c_out, h_out, w_out])
        }
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
        use crate::graph::shape::Expr;

        assert_eq!(
            self.view.ndim(),
            4,
            "conv3d: input must be 4D (C_in, D, H, W)"
        );
        assert_eq!(
            kernel.view.ndim(),
            5,
            "conv3d: kernel must be 5D (C_out, C_in/groups, kD, kH, kW)"
        );

        if groups == 1 {
            // 通常の畳み込み
            let kernel_d = match &kernel.view.shape()[2] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };
            let kernel_h = match &kernel.view.shape()[3] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };
            let kernel_w = match &kernel.view.shape()[4] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };

            // unfold: (C_in, D, H, W) -> (C_in, kD, kH, kW, D', H', W')
            let unfolded = self.unfold3d((kernel_d, kernel_h, kernel_w), stride, dilation, 1);

            // unfold: (C_in, kD, kH, kW, D', H', W') -> unsqueeze(0) -> (1, C_in, kD, kH, kW, D', H', W')
            let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(0));

            // kernel: (C_out, C_in, kD, kH, kW) -> unsqueeze -> (C_out, C_in, kD, kH, kW, 1, 1, 1)
            let kernel_tmp1 = kernel.view(kernel.view.clone().unsqueeze(5));
            let kernel_tmp2 = kernel_tmp1.view(kernel_tmp1.view.clone().unsqueeze(6));
            let kernel_expanded = kernel_tmp2.view(kernel_tmp2.view.clone().unsqueeze(7));

            // expand to common shape: (C_out, C_in, kD, kH, kW, D', H', W')
            let c_out = kernel.view.shape()[0].clone();
            let c_in = unfolded.view.shape()[0].clone();
            let kd = unfolded.view.shape()[1].clone();
            let kh = unfolded.view.shape()[2].clone();
            let kw = unfolded.view.shape()[3].clone();
            let d_out = unfolded.view.shape()[4].clone();
            let h_out = unfolded.view.shape()[5].clone();
            let w_out = unfolded.view.shape()[6].clone();
            let common_shape = vec![
                c_out.clone(),
                c_in.clone(),
                kd.clone(),
                kh.clone(),
                kw.clone(),
                d_out.clone(),
                h_out.clone(),
                w_out.clone(),
            ];

            let unfolded_broadcasted = unfolded_expanded.expand(common_shape.clone());
            let kernel_broadcasted = kernel_expanded.expand(common_shape);

            // mul: (C_out, C_in, kD, kH, kW, D', H', W')
            let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

            // reduce sum over C_in (axis=1), kD (axis=2), kH (axis=3), kW (axis=4)

            mul_result
                .reduce_sum(1)
                .reduce_sum(1)
                .reduce_sum(1)
                .reduce_sum(1)
        } else {
            // グループ畳み込み
            let c_in_per_group =
                match (self.view.shape()[0].clone() / Expr::from(groups as isize)).simplify() {
                    Expr::Const(c) => c as usize,
                    _ => panic!("C_in/groups must be constant"),
                };

            let kernel_d = match &kernel.view.shape()[2] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };
            let kernel_h = match &kernel.view.shape()[3] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };
            let kernel_w = match &kernel.view.shape()[4] {
                Expr::Const(c) => *c as usize,
                _ => panic!("kernel size must be constant"),
            };

            // unfold: (C_in, D, H, W) -> (groups, C_in/groups, kD, kH, kW, D', H', W')
            let unfolded = self.unfold3d((kernel_d, kernel_h, kernel_w), stride, dilation, groups);

            // unfold: (groups, C_in/groups, kD, kH, kW, D', H', W') -> unsqueeze(1) -> (groups, 1, C_in/groups, kD, kH, kW, D', H', W')
            let unfolded_expanded = unfolded.view(unfolded.view.clone().unsqueeze(1));

            // kernel: (C_out, C_in/groups, kD, kH, kW) を (groups, C_out/groups, C_in/groups, kD, kH, kW) にreshape
            // 注: reshapeの前にcontiguous化が必要（非連続Viewの場合に正しく動作しないため）
            let c_out = kernel.view.shape()[0].clone();
            let c_out_per_group = (c_out.clone() / Expr::from(groups as isize)).simplify();

            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.clone()],
                kernel_contiguous_view,
            );
            let kernel_reshaped = kernel_contiguous.reshape(vec![
                Expr::from(groups as isize),
                c_out_per_group.clone(),
                Expr::from(c_in_per_group as isize),
                Expr::from(kernel_d as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
            ]);

            // unsqueeze: (groups, C_out/groups, C_in/groups, kD, kH, kW, 1, 1, 1)
            let kernel_tmp1 = kernel_reshaped.view(kernel_reshaped.view.clone().unsqueeze(6));
            let kernel_tmp2 = kernel_tmp1.view(kernel_tmp1.view.clone().unsqueeze(7));
            let kernel_expanded = kernel_tmp2.view(kernel_tmp2.view.clone().unsqueeze(8));

            // expand to common shape: (groups, C_out/groups, C_in/groups, kD, kH, kW, D', H', W')
            let d_out = unfolded.view.shape()[5].clone();
            let h_out = unfolded.view.shape()[6].clone();
            let w_out = unfolded.view.shape()[7].clone();
            let common_shape = vec![
                Expr::from(groups as isize),
                c_out_per_group.clone(),
                Expr::from(c_in_per_group as isize),
                Expr::from(kernel_d as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
                d_out.clone(),
                h_out.clone(),
                w_out.clone(),
            ];

            let unfolded_broadcasted = unfolded_expanded.expand(common_shape.clone());
            let kernel_broadcasted = kernel_expanded.expand(common_shape);

            // mul: (groups, C_out/groups, C_in/groups, kD, kH, kW, D', H', W')
            let mul_result = &unfolded_broadcasted * &kernel_broadcasted;

            // reduce sum over C_in/groups (axis=2), kD (axis=3), kH (axis=4), kW (axis=5)
            let reduced = mul_result
                .reduce_sum(2)
                .reduce_sum(2)
                .reduce_sum(2)
                .reduce_sum(2);

            // reshape: (groups, C_out/groups, D', H', W') -> (C_out, D', H', W')
            let d_out = reduced.view.shape()[2].clone();
            let h_out = reduced.view.shape()[3].clone();
            let w_out = reduced.view.shape()[4].clone();

            reduced.reshape(vec![c_out, d_out, h_out, w_out])
        }
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
        use crate::graph::shape::Expr;

        assert_eq!(
            self.view.ndim(),
            2,
            "conv_transpose1d: input must be 2D (C_in, L)"
        );
        assert_eq!(
            kernel.view.ndim(),
            3,
            "conv_transpose1d: kernel must be 3D (C_in, C_out/groups, k)"
        );

        let input_shape = self.view.shape();
        let kernel_shape = kernel.view.shape();

        let c_in = input_shape[0].expect_usize("C_in must be constant");
        let l_in = input_shape[1].expect_usize("L_in must be constant");

        let kernel_c_in = kernel_shape[0].expect_usize("kernel C_in must be constant");
        let c_out_per_group = kernel_shape[1].expect_usize("C_out/groups must be constant");
        let kernel_size = kernel_shape[2].expect_usize("k must be constant");

        assert_eq!(
            c_in, kernel_c_in,
            "Input channels must match kernel input channels"
        );

        let c_out = c_out_per_group * groups;

        // 出力サイズを計算
        let l_out =
            (l_in - 1) * stride + dilation * (kernel_size - 1) + output_padding + 1 - 2 * padding;

        if groups == 1 {
            // === groups=1: 通常の転置畳み込み ===
            // 入力: (C_in, L_in)
            // カーネル: (C_in, C_out, k)

            // 入力をunsqueeze: (C_in, L_in) -> (C_in, 1, 1, L_in)
            let input_expanded_view = self.view.clone().unsqueeze(1).unsqueeze(2);
            let input_expanded = self.view(input_expanded_view);

            // カーネルをunsqueeze: (C_in, C_out, k) -> (C_in, C_out, k, 1)
            let kernel_expanded_view = kernel.view.clone().unsqueeze(3);
            let kernel_expanded = kernel.view(kernel_expanded_view);

            // 共通シェイプに展開: (C_in, C_out, k, L_in)
            let common_shape = vec![
                Expr::from(c_in as isize),
                Expr::from(c_out as isize),
                Expr::from(kernel_size as isize),
                Expr::from(l_in as isize),
            ];

            let input_bc = input_expanded.expand(common_shape.clone());
            let kernel_bc = kernel_expanded.expand(common_shape);

            // 乗算: (C_in, C_out, k, L_in)
            let multiplied = &input_bc * &kernel_bc;

            // C_in軸でreduce: (C_out, k, L_in)
            let reduced = multiplied.reduce_sum(0);

            // reshape for fold: (C_out, k, L_in) -> (1, C_out * k, L_in)
            let reshaped_view = reduced.view.clone().reshape(vec![
                Expr::from(1),
                Expr::from((c_out * kernel_size) as isize),
                Expr::from(l_in as isize),
            ]);
            // contiguous化してからreshape
            let reduced_cont_view =
                crate::graph::shape::View::contiguous(reduced.view.shape().to_vec());
            let reduced_cont = crate::graph::GraphNode::new(
                reduced.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![reduced.clone()],
                reduced_cont_view,
            );
            let reshaped = reduced_cont.view(reshaped_view);

            // fold: 出力形状に畳み込む
            let folded = reshaped.fold1d(vec![c_out, l_out], kernel_size, stride, dilation, 1);

            // paddingがある場合はスライスで削る
            if padding > 0 {
                let l_start = padding;
                let l_end = l_out + 2 * padding - padding;
                folded.slice(vec![(0, c_out), (l_start, l_end)])
            } else {
                folded
            }
        } else {
            // === groups>1: グループ転置畳み込み ===
            let c_in_per_group = c_in / groups;

            // 入力をreshape: (C_in, L_in) -> (groups, C_in/groups, L_in)
            let input_contiguous_view =
                crate::graph::shape::View::contiguous(self.view.shape().to_vec());
            let input_contiguous = crate::graph::GraphNode::new(
                self.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![self.clone()],
                input_contiguous_view,
            );
            let input_reshaped_view = input_contiguous.view.clone().reshape(vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(l_in as isize),
            ]);
            let input_reshaped = input_contiguous.view(input_reshaped_view);

            // 入力をunsqueeze: (groups, C_in/groups, L_in) -> (groups, C_in/groups, 1, 1, L_in)
            let input_expanded =
                input_reshaped.view(input_reshaped.view.clone().unsqueeze(2).unsqueeze(3));

            // カーネルをreshape: (C_in, C_out/groups, k) -> (groups, C_in/groups, C_out/groups, k)
            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.clone()],
                kernel_contiguous_view,
            );
            let kernel_reshaped_view = kernel_contiguous.view.clone().reshape(vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(c_out_per_group as isize),
                Expr::from(kernel_size as isize),
            ]);
            let kernel_reshaped = kernel_contiguous.view(kernel_reshaped_view);

            // カーネルをunsqueeze: (groups, C_in/groups, C_out/groups, k) -> (groups, C_in/groups, C_out/groups, k, 1)
            let kernel_expanded = kernel_reshaped.view(kernel_reshaped.view.clone().unsqueeze(4));

            // 共通シェイプに展開: (groups, C_in/groups, C_out/groups, k, L_in)
            let common_shape = vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(c_out_per_group as isize),
                Expr::from(kernel_size as isize),
                Expr::from(l_in as isize),
            ];

            let input_bc = input_expanded.expand(common_shape.clone());
            let kernel_bc = kernel_expanded.expand(common_shape);

            // 乗算: (groups, C_in/groups, C_out/groups, k, L_in)
            let multiplied = &input_bc * &kernel_bc;

            // C_in/groups軸でreduce: (groups, C_out/groups, k, L_in)
            let reduced = multiplied.reduce_sum(1);

            // fold with groups
            let folded = reduced.fold1d(vec![c_out, l_out], kernel_size, stride, dilation, groups);

            // paddingがある場合はスライスで削る
            if padding > 0 {
                let l_start = padding;
                let l_end = l_out + 2 * padding - padding;
                folded.slice(vec![(0, c_out), (l_start, l_end)])
            } else {
                folded
            }
        }
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
        use crate::graph::shape::Expr;

        assert_eq!(
            self.view.ndim(),
            3,
            "conv_transpose2d: input must be 3D (C_in, H, W)"
        );
        assert_eq!(
            kernel.view.ndim(),
            4,
            "conv_transpose2d: kernel must be 4D (C_in, C_out/groups, kH, kW)"
        );

        let input_shape = self.view.shape();
        let kernel_shape = kernel.view.shape();

        let c_in = input_shape[0].expect_usize("C_in must be constant");
        let h_in = input_shape[1].expect_usize("H_in must be constant");
        let w_in = input_shape[2].expect_usize("W_in must be constant");

        let kernel_c_in = kernel_shape[0].expect_usize("kernel C_in must be constant");
        let c_out_per_group = kernel_shape[1].expect_usize("C_out/groups must be constant");
        let kernel_h = kernel_shape[2].expect_usize("kH must be constant");
        let kernel_w = kernel_shape[3].expect_usize("kW must be constant");

        assert_eq!(
            c_in, kernel_c_in,
            "Input channels must match kernel input channels"
        );

        let c_out = c_out_per_group * groups;

        // 出力サイズを計算
        let h_out = (h_in - 1) * stride.0 + dilation.0 * (kernel_h - 1) + output_padding.0 + 1
            - 2 * padding.0;
        let w_out = (w_in - 1) * stride.1 + dilation.1 * (kernel_w - 1) + output_padding.1 + 1
            - 2 * padding.1;

        if groups == 1 {
            // === groups=1: 通常の転置畳み込み ===
            // 入力: (C_in, H_in, W_in)
            // カーネル: (C_in, C_out, kH, kW)

            // 入力をunsqueeze: (C_in, H_in, W_in) -> (C_in, 1, 1, 1, H_in, W_in)
            let input_expanded_view = self.view.clone().unsqueeze(1).unsqueeze(2).unsqueeze(3);
            let input_expanded = self.view(input_expanded_view);

            // カーネルをunsqueeze: (C_in, C_out, kH, kW) -> (C_in, C_out, kH, kW, 1, 1)
            let kernel_expanded_view = kernel.view.clone().unsqueeze(4).unsqueeze(5);
            let kernel_expanded = kernel.view(kernel_expanded_view);

            // 共通シェイプに展開: (C_in, C_out, kH, kW, H_in, W_in)
            let common_shape = vec![
                Expr::from(c_in as isize),
                Expr::from(c_out as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
                Expr::from(h_in as isize),
                Expr::from(w_in as isize),
            ];

            let input_bc = input_expanded.expand(common_shape.clone());
            let kernel_bc = kernel_expanded.expand(common_shape);

            // 乗算: (C_in, C_out, kH, kW, H_in, W_in)
            let multiplied = &input_bc * &kernel_bc;

            // C_in軸でreduce: (C_out, kH, kW, H_in, W_in)
            let reduced = multiplied.reduce_sum(0);

            // reshape for fold: (C_out, kH, kW, H_in, W_in) -> (1, C_out * kH * kW, H_in * W_in)
            let reshaped_view = reduced.view.clone().reshape(vec![
                Expr::from(1),
                Expr::from((c_out * kernel_h * kernel_w) as isize),
                Expr::from((h_in * w_in) as isize),
            ]);
            // contiguous化してからreshape
            let reduced_cont_view =
                crate::graph::shape::View::contiguous(reduced.view.shape().to_vec());
            let reduced_cont = crate::graph::GraphNode::new(
                reduced.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![reduced.clone()],
                reduced_cont_view,
            );
            let reshaped = reduced_cont.view(reshaped_view);

            // fold: 出力形状に畳み込む
            let folded = reshaped.fold2d(
                vec![c_out, h_out, w_out],
                (kernel_h, kernel_w),
                stride,
                dilation,
                1,
            );

            // paddingがある場合はスライスで削る
            if padding.0 > 0 || padding.1 > 0 {
                let h_start = padding.0;
                let h_end = h_out + 2 * padding.0 - padding.0;
                let w_start = padding.1;
                let w_end = w_out + 2 * padding.1 - padding.1;
                folded.slice(vec![(0, c_out), (h_start, h_end), (w_start, w_end)])
            } else {
                folded
            }
        } else {
            // === groups>1: グループ転置畳み込み ===
            let c_in_per_group = c_in / groups;

            // 入力をreshape: (C_in, H_in, W_in) -> (groups, C_in/groups, H_in, W_in)
            let input_contiguous_view =
                crate::graph::shape::View::contiguous(self.view.shape().to_vec());
            let input_contiguous = crate::graph::GraphNode::new(
                self.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![self.clone()],
                input_contiguous_view,
            );
            let input_reshaped_view = input_contiguous.view.clone().reshape(vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(h_in as isize),
                Expr::from(w_in as isize),
            ]);
            let input_reshaped = input_contiguous.view(input_reshaped_view);

            // 入力をunsqueeze: (groups, C_in/groups, H_in, W_in) -> (groups, C_in/groups, 1, 1, 1, H_in, W_in)
            let input_expanded = input_reshaped.view(
                input_reshaped
                    .view
                    .clone()
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .unsqueeze(4),
            );

            // カーネルをreshape: (C_in, C_out/groups, kH, kW) -> (groups, C_in/groups, C_out/groups, kH, kW)
            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.clone()],
                kernel_contiguous_view,
            );
            let kernel_reshaped_view = kernel_contiguous.view.clone().reshape(vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(c_out_per_group as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
            ]);
            let kernel_reshaped = kernel_contiguous.view(kernel_reshaped_view);

            // カーネルをunsqueeze: (groups, C_in/groups, C_out/groups, kH, kW) -> (groups, C_in/groups, C_out/groups, kH, kW, 1, 1)
            let kernel_expanded =
                kernel_reshaped.view(kernel_reshaped.view.clone().unsqueeze(5).unsqueeze(6));

            // 共通シェイプに展開: (groups, C_in/groups, C_out/groups, kH, kW, H_in, W_in)
            let common_shape = vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(c_out_per_group as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
                Expr::from(h_in as isize),
                Expr::from(w_in as isize),
            ];

            let input_bc = input_expanded.expand(common_shape.clone());
            let kernel_bc = kernel_expanded.expand(common_shape);

            // 乗算: (groups, C_in/groups, C_out/groups, kH, kW, H_in, W_in)
            let multiplied = &input_bc * &kernel_bc;

            // C_in/groups軸でreduce: (groups, C_out/groups, kH, kW, H_in, W_in)
            let reduced = multiplied.reduce_sum(1);

            // fold with groups
            let folded = reduced.fold2d(
                vec![c_out, h_out, w_out],
                (kernel_h, kernel_w),
                stride,
                dilation,
                groups,
            );

            // paddingがある場合はスライスで削る
            if padding.0 > 0 || padding.1 > 0 {
                let h_start = padding.0;
                let h_end = h_out + 2 * padding.0 - padding.0;
                let w_start = padding.1;
                let w_end = w_out + 2 * padding.1 - padding.1;
                folded.slice(vec![(0, c_out), (h_start, h_end), (w_start, w_end)])
            } else {
                folded
            }
        }
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
        use crate::graph::shape::Expr;

        assert_eq!(
            self.view.ndim(),
            4,
            "conv_transpose3d: input must be 4D (C_in, D, H, W)"
        );
        assert_eq!(
            kernel.view.ndim(),
            5,
            "conv_transpose3d: kernel must be 5D (C_in, C_out/groups, kD, kH, kW)"
        );

        let input_shape = self.view.shape();
        let kernel_shape = kernel.view.shape();

        let c_in = input_shape[0].expect_usize("C_in must be constant");
        let d_in = input_shape[1].expect_usize("D_in must be constant");
        let h_in = input_shape[2].expect_usize("H_in must be constant");
        let w_in = input_shape[3].expect_usize("W_in must be constant");

        let kernel_c_in = kernel_shape[0].expect_usize("kernel C_in must be constant");
        let c_out_per_group = kernel_shape[1].expect_usize("C_out/groups must be constant");
        let kernel_d = kernel_shape[2].expect_usize("kD must be constant");
        let kernel_h = kernel_shape[3].expect_usize("kH must be constant");
        let kernel_w = kernel_shape[4].expect_usize("kW must be constant");

        assert_eq!(
            c_in, kernel_c_in,
            "Input channels must match kernel input channels"
        );

        let c_out = c_out_per_group * groups;

        // 出力サイズを計算
        let d_out = (d_in - 1) * stride.0 + dilation.0 * (kernel_d - 1) + output_padding.0 + 1
            - 2 * padding.0;
        let h_out = (h_in - 1) * stride.1 + dilation.1 * (kernel_h - 1) + output_padding.1 + 1
            - 2 * padding.1;
        let w_out = (w_in - 1) * stride.2 + dilation.2 * (kernel_w - 1) + output_padding.2 + 1
            - 2 * padding.2;

        if groups == 1 {
            // === groups=1: 通常の転置畳み込み ===
            // 入力: (C_in, D_in, H_in, W_in)
            // カーネル: (C_in, C_out, kD, kH, kW)

            // 入力をunsqueeze: (C_in, D_in, H_in, W_in) -> (C_in, 1, 1, 1, 1, D_in, H_in, W_in)
            let input_expanded_view = self
                .view
                .clone()
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4);
            let input_expanded = self.view(input_expanded_view);

            // カーネルをunsqueeze: (C_in, C_out, kD, kH, kW) -> (C_in, C_out, kD, kH, kW, 1, 1, 1)
            let kernel_expanded_view = kernel.view.clone().unsqueeze(5).unsqueeze(6).unsqueeze(7);
            let kernel_expanded = kernel.view(kernel_expanded_view);

            // 共通シェイプに展開: (C_in, C_out, kD, kH, kW, D_in, H_in, W_in)
            let common_shape = vec![
                Expr::from(c_in as isize),
                Expr::from(c_out as isize),
                Expr::from(kernel_d as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
                Expr::from(d_in as isize),
                Expr::from(h_in as isize),
                Expr::from(w_in as isize),
            ];

            let input_bc = input_expanded.expand(common_shape.clone());
            let kernel_bc = kernel_expanded.expand(common_shape);

            // 乗算: (C_in, C_out, kD, kH, kW, D_in, H_in, W_in)
            let multiplied = &input_bc * &kernel_bc;

            // C_in軸でreduce: (C_out, kD, kH, kW, D_in, H_in, W_in)
            let reduced = multiplied.reduce_sum(0);

            // reshape for fold: (C_out, kD, kH, kW, D_in, H_in, W_in) -> (1, C_out * kD * kH * kW, D_in * H_in * W_in)
            let reshaped_view = reduced.view.clone().reshape(vec![
                Expr::from(1),
                Expr::from((c_out * kernel_d * kernel_h * kernel_w) as isize),
                Expr::from((d_in * h_in * w_in) as isize),
            ]);
            // contiguous化してからreshape
            let reduced_cont_view =
                crate::graph::shape::View::contiguous(reduced.view.shape().to_vec());
            let reduced_cont = crate::graph::GraphNode::new(
                reduced.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![reduced.clone()],
                reduced_cont_view,
            );
            let reshaped = reduced_cont.view(reshaped_view);

            // fold: 出力形状に畳み込む
            let folded = reshaped.fold3d(
                vec![c_out, d_out, h_out, w_out],
                (kernel_d, kernel_h, kernel_w),
                stride,
                dilation,
                1,
            );

            // paddingがある場合はスライスで削る
            if padding.0 > 0 || padding.1 > 0 || padding.2 > 0 {
                let d_start = padding.0;
                let d_end = d_out + 2 * padding.0 - padding.0;
                let h_start = padding.1;
                let h_end = h_out + 2 * padding.1 - padding.1;
                let w_start = padding.2;
                let w_end = w_out + 2 * padding.2 - padding.2;
                folded.slice(vec![
                    (0, c_out),
                    (d_start, d_end),
                    (h_start, h_end),
                    (w_start, w_end),
                ])
            } else {
                folded
            }
        } else {
            // === groups>1: グループ転置畳み込み ===
            let c_in_per_group = c_in / groups;

            // 入力をreshape: (C_in, D_in, H_in, W_in) -> (groups, C_in/groups, D_in, H_in, W_in)
            let input_contiguous_view =
                crate::graph::shape::View::contiguous(self.view.shape().to_vec());
            let input_contiguous = crate::graph::GraphNode::new(
                self.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![self.clone()],
                input_contiguous_view,
            );
            let input_reshaped_view = input_contiguous.view.clone().reshape(vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(d_in as isize),
                Expr::from(h_in as isize),
                Expr::from(w_in as isize),
            ]);
            let input_reshaped = input_contiguous.view(input_reshaped_view);

            // 入力をunsqueeze: (groups, C_in/groups, D_in, H_in, W_in) -> (groups, C_in/groups, 1, 1, 1, 1, D_in, H_in, W_in)
            let input_expanded = input_reshaped.view(
                input_reshaped
                    .view
                    .clone()
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .unsqueeze(4)
                    .unsqueeze(5),
            );

            // カーネルをreshape: (C_in, C_out/groups, kD, kH, kW) -> (groups, C_in/groups, C_out/groups, kD, kH, kW)
            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.clone()],
                kernel_contiguous_view,
            );
            let kernel_reshaped_view = kernel_contiguous.view.clone().reshape(vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(c_out_per_group as isize),
                Expr::from(kernel_d as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
            ]);
            let kernel_reshaped = kernel_contiguous.view(kernel_reshaped_view);

            // カーネルをunsqueeze: (groups, C_in/groups, C_out/groups, kD, kH, kW) -> (groups, C_in/groups, C_out/groups, kD, kH, kW, 1, 1, 1)
            let kernel_expanded = kernel_reshaped.view(
                kernel_reshaped
                    .view
                    .clone()
                    .unsqueeze(6)
                    .unsqueeze(7)
                    .unsqueeze(8),
            );

            // 共通シェイプに展開: (groups, C_in/groups, C_out/groups, kD, kH, kW, D_in, H_in, W_in)
            let common_shape = vec![
                Expr::from(groups as isize),
                Expr::from(c_in_per_group as isize),
                Expr::from(c_out_per_group as isize),
                Expr::from(kernel_d as isize),
                Expr::from(kernel_h as isize),
                Expr::from(kernel_w as isize),
                Expr::from(d_in as isize),
                Expr::from(h_in as isize),
                Expr::from(w_in as isize),
            ];

            let input_bc = input_expanded.expand(common_shape.clone());
            let kernel_bc = kernel_expanded.expand(common_shape);

            // 乗算: (groups, C_in/groups, C_out/groups, kD, kH, kW, D_in, H_in, W_in)
            let multiplied = &input_bc * &kernel_bc;

            // C_in/groups軸でreduce: (groups, C_out/groups, kD, kH, kW, D_in, H_in, W_in)
            let reduced = multiplied.reduce_sum(1);

            // fold with groups
            let folded = reduced.fold3d(
                vec![c_out, d_out, h_out, w_out],
                (kernel_d, kernel_h, kernel_w),
                stride,
                dilation,
                groups,
            );

            // paddingがある場合はスライスで削る
            if padding.0 > 0 || padding.1 > 0 || padding.2 > 0 {
                let d_start = padding.0;
                let d_end = d_out + 2 * padding.0 - padding.0;
                let h_start = padding.1;
                let h_end = h_out + 2 * padding.1 - padding.1;
                let w_start = padding.2;
                let w_end = w_out + 2 * padding.2 - padding.2;
                folded.slice(vec![
                    (0, c_out),
                    (d_start, d_end),
                    (h_start, h_end),
                    (w_start, w_end),
                ])
            } else {
                folded
            }
        }
    }
}
