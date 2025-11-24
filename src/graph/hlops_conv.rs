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
            let c_out = kernel.view.shape()[0].clone();
            let c_out_per_group = (c_out.clone() / Expr::from(groups as isize)).simplify();

            let kernel_reshaped = kernel.reshape(vec![
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
            let c_out = kernel.view.shape()[0].clone();
            let c_out_per_group = (c_out.clone() / Expr::from(groups as isize)).simplify();

            let kernel_reshaped = kernel.reshape(vec![
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
            let c_out = kernel.view.shape()[0].clone();
            let c_out_per_group = (c_out.clone() / Expr::from(groups as isize)).simplify();

            let kernel_reshaped = kernel.reshape(vec![
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
}
