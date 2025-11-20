//! 高レベル演算のヘルパー関数
//!
//! このモジュールは既存の基本的なグラフ演算を組み合わせて、
//! より高レベルな数学的演算や便利な演算を提供します。

use crate::graph::GraphNode;
use crate::graph::ops::{ElementwiseOp, GraphOp, max, recip, reduce_sum};

impl GraphNode {
    /// 二乗: x^2
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![10, 20])
    ///     .build();
    ///
    /// let x_squared = x.square();
    /// ```
    pub fn square(self) -> GraphNode {
        &self * &self
    }

    /// 累乗: x^n (正の整数のみ)
    ///
    /// # パニック
    /// n が 0 の場合はパニックします
    pub fn powi(self, n: u32) -> GraphNode {
        assert!(n > 0, "powi: n must be positive");

        if n == 1 {
            return self;
        }

        let mut result = self.clone();
        for _ in 1..n {
            result = &result * &self;
        }
        result
    }

    /// 絶対値の二乗: x^2 (常に非負)
    ///
    /// Note: 本物の絶対値を実装するには max(x, -x) が必要ですが、
    /// 現在の実装では異なるテンソル間の要素ごとの最大値しかサポートしていないため、
    /// 代わりに二乗を使って非負の値を得ることができます。
    pub fn abs_square(self) -> GraphNode {
        self.square()
    }

    /// 2つのテンソルの要素ごとの最小値: min(a, b) = -max(-a, -b)
    pub fn min(self, other: GraphNode) -> GraphNode {
        -max(-self, -other)
    }

    /// クランプ: min_val <= x <= max_val に制限
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![10])
    ///     .build();
    /// let min_val = graph.input("min")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![10])
    ///     .build();
    /// let max_val = graph.input("max")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![10])
    ///     .build();
    ///
    /// let clamped = x.clamp(min_val, max_val);
    /// ```
    pub fn clamp(self, min_val: GraphNode, max_val: GraphNode) -> GraphNode {
        max(self, min_val).min(max_val)
    }

    /// 平均を計算: mean(x, axis)
    ///
    /// 指定された軸に沿った平均を計算します。
    /// sum(x, axis) / size(axis) として実装されます。
    ///
    /// Note: 現在の実装では、軸のサイズが定数（Expr::Const）の場合のみサポートされます。
    /// 変数サイズの軸に対してはパニックします。
    pub fn mean(self, axis: usize) -> GraphNode {
        use crate::graph::shape::Expr;

        let shape = self.view.shape();
        if axis >= shape.len() {
            panic!("mean: axis {} is out of bounds for shape {:?}", axis, shape);
        }

        // 軸のサイズを取得
        let axis_size = &shape[axis];

        // 軸サイズが定数の場合のみ処理
        let size_value = match axis_size {
            Expr::Const(n) => *n as f32,
            _ => panic!(
                "mean: axis size must be constant, got symbolic expression: {:?}",
                axis_size
            ),
        };

        // 合計を計算し、サイズで割る
        // スカラーは自動的にブロードキャストされる
        reduce_sum(self, axis) * (1.0f32 / size_value)
    }

    /// 分散を計算: var(x, axis)
    ///
    /// 不偏分散を計算します: E[(x - mean(x))^2]
    ///
    /// Note: この実装は効率的ではありません（meanを2回計算）。
    /// より効率的な実装には E[x^2] - E[x]^2 を使用できますが、
    /// 数値安定性の問題があります。
    pub fn variance(self, axis: usize) -> GraphNode {
        // 平均を計算
        let x_mean = self.clone().mean(axis);

        // meanの次元を復元してbroadcast可能にする
        let x_mean_expanded = x_mean.view(
            x_mean
                .view
                .clone()
                .unsqueeze(axis)
                .expand(self.view.shape().to_vec()),
        );

        // (x - mean)^2 の平均を計算
        (self - x_mean_expanded).square().mean(axis)
    }

    /// 行列積 (2次元テンソル限定): C[i,j] = sum_k(A[i,k] * B[k,j])
    ///
    /// # パニック
    /// - a が 2次元テンソルでない場合
    /// - b が 2次元テンソルでない場合
    /// - a の列数と b の行数が一致しない場合
    ///
    /// # 実装の詳細
    /// 1. a を [i, k] -> [i, k, 1] に reshape (unsqueeze)
    /// 2. b を [k, j] -> [1, k, j] に reshape (unsqueeze)
    /// 3. a と b を broadcast して [i, k, j] にする
    /// 4. 要素ごとの積を取る: [i, k, j]
    /// 5. k 軸で reduce_sum: [i, j]
    ///
    /// Note: 現在の実装ではbroadcastとunsqueezeが必要なため、
    /// 将来的な実装として残します。
    pub fn matmul(self, other: GraphNode) -> GraphNode {
        let a_shape = self.view.shape();
        let b_shape = other.view.shape();

        // 2次元チェック
        assert_eq!(a_shape.len(), 2, "matmul: first argument must be 2D");
        assert_eq!(b_shape.len(), 2, "matmul: second argument must be 2D");

        // 次元の互換性チェック
        assert_eq!(
            a_shape[1], b_shape[0],
            "matmul: incompatible dimensions: {:?} and {:?}",
            a_shape, b_shape
        );

        // TODO: unsqueeze, expand, broadcastの実装が必要
        // 現在は実装を保留
        panic!("matmul: not yet implemented - requires unsqueeze and broadcast operations");
    }

    /// バッチ行列積 (3次元テンソル): C[b,i,j] = sum_k(A[b,i,k] * B[b,k,j])
    ///
    /// バッチ次元を持つ行列積を計算します。
    ///
    /// # パニック
    /// - a が 3次元テンソルでない場合
    /// - b が 3次元テンソルでない場合
    /// - バッチサイズが一致しない場合
    /// - 行列の次元が互換性がない場合
    pub fn batch_matmul(self, other: GraphNode) -> GraphNode {
        let a_shape = self.view.shape();
        let b_shape = other.view.shape();

        // 3次元チェック
        assert_eq!(a_shape.len(), 3, "batch_matmul: first argument must be 3D");
        assert_eq!(b_shape.len(), 3, "batch_matmul: second argument must be 3D");

        // バッチサイズチェック
        assert_eq!(
            a_shape[0], b_shape[0],
            "batch_matmul: batch sizes must match: {:?} vs {:?}",
            a_shape[0], b_shape[0]
        );

        // 行列次元の互換性チェック
        assert_eq!(
            a_shape[2], b_shape[1],
            "batch_matmul: incompatible matrix dimensions: {:?} and {:?}",
            a_shape, b_shape
        );

        // TODO: unsqueeze, expand, broadcastの実装が必要
        panic!("batch_matmul: not yet implemented - requires unsqueeze and broadcast operations");
    }

    // ============================================================================
    // 基本的な数学関数
    // ============================================================================

    /// 底が2の対数: log2(x)
    pub fn log2(self) -> GraphNode {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Log2,
                elementwise_strategies: None,
            },
            vec![self],
            view,
        )
    }

    /// 2の累乗: 2^x
    pub fn exp2(self) -> GraphNode {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Exp2,
                elementwise_strategies: None,
            },
            vec![self],
            view,
        )
    }

    /// 自然対数: ln(x) = log(x)
    ///
    /// log2を使って実装: log(x) = log2(x) / log2(e)
    pub fn log(self) -> GraphNode {
        // log(x) = log2(x) * (1 / log2(e))
        // 1 / log2(e) ≈ 0.6931471805599453
        const INV_LOG2_E: f32 = 1.0 / std::f32::consts::LOG2_E;

        // スカラー定数は自動的にブロードキャストされる
        self.log2() * INV_LOG2_E
    }

    /// 指数関数: e^x = exp(x)
    ///
    /// exp2を使って実装: exp(x) = 2^(x * log2(e))
    pub fn exp(self) -> GraphNode {
        // exp(x) = 2^(x * log2(e))
        const LOG2_E: f32 = std::f32::consts::LOG2_E;

        // スカラー定数は自動的にブロードキャストされる
        (self * LOG2_E).exp2()
    }

    /// 正弦: sin(x)
    pub fn sin(self) -> GraphNode {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Sin,
                elementwise_strategies: None,
            },
            vec![self],
            view,
        )
    }

    /// 余弦: cos(x) = sin(x + π/2)
    ///
    /// sinを使って実装します。
    pub fn cos(self) -> GraphNode {
        // cos(x) = sin(x + π/2)
        const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;

        // スカラー定数は自動的にブロードキャストされる
        (self + HALF_PI).sin()
    }

    /// 平方根: sqrt(x)
    pub fn sqrt(self) -> GraphNode {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Sqrt,
                elementwise_strategies: None,
            },
            vec![self],
            view,
        )
    }

    /// 平方根の逆数: rsqrt(x) = 1/sqrt(x)
    ///
    /// sqrtとrecipを使って実装します。
    pub fn rsqrt(self) -> GraphNode {
        recip(self.sqrt())
    }

    /// 1D畳み込み
    ///
    /// unfold、elementwise乗算、reduceを組み合わせて畳み込みを実装します。
    ///
    /// # 引数
    /// - `kernel`: 畳み込みカーネル (C_out, C_in/groups, k)
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数（1=通常、C_in=depthwise）
    ///
    /// # 入出力
    /// - 入力: (C_in, L)
    /// - カーネル: (C_out, C_in/groups, k)
    /// - 出力: (C_out, L')
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 32])
    ///     .build();
    /// let kernel = graph.input("kernel")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![16, 3, 3])
    ///     .build();
    ///
    /// // (3, 32) conv (16, 3, 3) -> (16, 30)
    /// let output = x.conv1d(kernel, 1, 1, 1);
    /// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::shape::Expr;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_square() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let x_squared = x.square();

        // 型とshapeが保持されていることを確認
        match x_squared.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(x_squared.view.ndim(), 2);
        assert_eq!(x_squared.view.shape()[0], Expr::from(10));
        assert_eq!(x_squared.view.shape()[1], Expr::from(20));
    }

    #[test]
    fn test_cube() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5])
            .build();

        let x_cubed = x.powi(3);

        match x_cubed.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(x_cubed.view.ndim(), 1);
    }

    #[test]
    fn test_powi() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5])
            .build();

        let x_pow4 = x.powi(4);

        match x_pow4.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(x_pow4.view.ndim(), 1);
    }

    #[test]
    #[should_panic(expected = "powi: n must be positive")]
    fn test_powi_zero() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5])
            .build();

        let _ = x.powi(0);
    }

    #[test]
    fn test_abs_square() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let abs_sq = x.abs_square();

        match abs_sq.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(abs_sq.view.ndim(), 1);
    }

    #[test]
    fn test_min() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let min_ab = a.min(b);

        match min_ab.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(min_ab.view.ndim(), 1);
    }

    #[test]
    fn test_clamp() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let min_val = graph
            .input("min")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let max_val = graph
            .input("max")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let clamped = x.clamp(min_val, max_val);

        match clamped.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(clamped.view.ndim(), 1);
    }

    #[test]
    fn test_mean() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20, 30])
            .build();

        // Note: 現在の実装では実際にはsumのみを計算
        let mean_x = x.mean(1);

        match mean_x.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        // 軸1が縮約されている
        assert_eq!(mean_x.view.ndim(), 2);
    }

    #[test]
    #[should_panic(expected = "mean: axis")]
    fn test_mean_out_of_bounds() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let _ = x.mean(3);
    }

    #[test]
    #[should_panic(expected = "matmul: not yet implemented")]
    fn test_matmul_not_implemented() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![20, 30])
            .build();

        let _ = a.matmul(b);
    }

    #[test]
    #[should_panic(expected = "matmul: first argument must be 2D")]
    fn test_matmul_wrong_dim_a() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let _ = a.matmul(b);
    }

    #[test]
    #[should_panic(expected = "batch_matmul: not yet implemented")]
    fn test_batch_matmul_not_implemented() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 10, 20])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 20, 30])
            .build();

        let _ = a.batch_matmul(b);
    }

    #[test]
    fn test_log2() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = x.log2();

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);

        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Log2,
                ..
            } => {}
            _ => panic!("Expected Log2 operation"),
        }
    }

    #[test]
    fn test_exp2() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 5])
            .build();

        let result = x.exp2();

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 2);

        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Exp2,
                ..
            } => {}
            _ => panic!("Expected Exp2 operation"),
        }
    }

    #[test]
    fn test_log() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = x.log();

        // log(x) = log2(x) * const なので、最終的にMul演算になる
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
    }

    #[test]
    fn test_exp() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = x.exp();

        // exp(x) = exp2(x * const) なので、exp2演算が含まれる
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
    }

    #[test]
    fn test_sin() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = x.sin();

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);

        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Sin,
                ..
            } => {}
            _ => panic!("Expected Sin operation"),
        }
    }

    #[test]
    fn test_cos() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = x.cos();

        // cos(x) = sin(x + const)
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
    }

    #[test]
    fn test_sqrt() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = x.sqrt();

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);

        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Sqrt,
                ..
            } => {}
            _ => panic!("Expected Sqrt operation"),
        }
    }

    #[test]
    fn test_rsqrt() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = x.rsqrt();

        // rsqrt(x) = recip(sqrt(x))
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
    }
}
