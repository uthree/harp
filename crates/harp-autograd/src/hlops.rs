//! 高レベル演算のヘルパー関数
//!
//! このモジュールは既存の基本的な autograd 演算を組み合わせて、
//! より高レベルな数学的演算や便利な演算を提供します。
//!
//! graphモジュールのhlops.rsと同じ設計思想で、
//! 他の演算の組み合わせで実現できる演算をここに配置します。
//! ノード数が増えることは後段の最適化機能で対処します。

use super::Tensor;

impl Tensor {
    // ============================================================================
    // 基本的な数学関数（他の演算の組み合わせで実現）
    // ============================================================================

    /// 自然対数: ln(x) = log(x)
    ///
    /// log2を使って実装: log(x) = log2(x) / log2(e)
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![10]);
    /// let log_x = x.log();
    /// ```
    pub fn log(&self) -> Tensor {
        // log(x) = log2(x) * (1 / log2(e))
        // 1 / log2(e) ≈ 0.6931471805599453
        const INV_LOG2_E: f32 = 1.0 / std::f32::consts::LOG2_E;
        &self.log2() * INV_LOG2_E
    }

    /// 指数関数: e^x = exp(x)
    ///
    /// exp2を使って実装: exp(x) = 2^(x * log2(e))
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::zeros(vec![5]);
    /// let exp_x = x.exp();
    /// ```
    pub fn exp(&self) -> Tensor {
        // exp(x) = 2^(x * log2(e))
        const LOG2_E: f32 = std::f32::consts::LOG2_E;
        (self * LOG2_E).exp2()
    }

    /// 余弦: cos(x) = sin(x + π/2)
    ///
    /// sinを使って実装します。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::zeros(vec![10]);
    /// let cos_x = x.cos();
    /// ```
    pub fn cos(&self) -> Tensor {
        // cos(x) = sin(x + π/2)
        const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;
        (self + HALF_PI).sin()
    }

    /// 平方根の逆数: rsqrt(x) = 1/sqrt(x)
    ///
    /// sqrtとrecipを使って実装します。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![5]);
    /// let rsqrt_x = x.rsqrt();
    /// ```
    pub fn rsqrt(&self) -> Tensor {
        self.sqrt().recip()
    }

    // ============================================================================
    // 代数演算
    // ============================================================================

    /// 二乗: x^2
    ///
    /// 乗算を使って実装します。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![3, 4]);
    /// let x_squared = x.square();
    /// ```
    pub fn square(&self) -> Tensor {
        self * self
    }

    /// 累乗: x^n (正の整数のみ)
    ///
    /// 乗算を繰り返して実装します。
    ///
    /// # パニック
    /// n が 0 の場合はパニックします
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![5]);
    /// let x_cubed = x.powi(3);
    /// ```
    pub fn powi(&self, n: u32) -> Tensor {
        assert!(n > 0, "powi: n must be positive");

        if n == 1 {
            return self.clone();
        }

        let mut result = self.clone();
        for _ in 1..n {
            result = &result * self;
        }
        result
    }

    /// 絶対値の二乗: x^2 (常に非負)
    ///
    /// Note: 本物の絶対値を実装するには max(x, -x) が必要ですが、
    /// 現在の実装では異なるテンソル間の要素ごとの最大値しかサポートしていないため、
    /// 代わりに二乗を使って非負の値を得ることができます。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::ones(vec![10]);
    /// let abs_sq = x.abs_square();
    /// ```
    pub fn abs_square(&self) -> Tensor {
        self.square()
    }

    /// 2つのテンソルの要素ごとの最小値: min(a, b) = -max(-a, -b)
    ///
    /// maxと符号反転を使って実装します。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let a = Tensor::ones(vec![10]);
    /// let b = Tensor::zeros(vec![10]);
    /// let min_ab = a.min(&b);
    /// ```
    pub fn min(&self, other: &Tensor) -> Tensor {
        let neg_self = -self;
        let neg_other = -other;
        -neg_self.max(&neg_other)
    }

    /// クランプ: min_val <= x <= max_val に制限
    ///
    /// maxとminを使って実装します。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    ///
    /// let x = Tensor::full(vec![10], 5.0);
    /// let min_val = Tensor::zeros(vec![10]);
    /// let max_val = Tensor::ones(vec![10]);
    /// let clamped = x.clamp(&min_val, &max_val);
    /// ```
    pub fn clamp(&self, min_val: &Tensor, max_val: &Tensor) -> Tensor {
        self.max(min_val).min(max_val)
    }

    // ============================================================================
    // 統計演算
    // ============================================================================

    /// 平均を計算: mean(x, axis)
    ///
    /// 指定された軸に沿った平均を計算します。
    /// sum(x, axis) / size(axis) として実装されます。
    ///
    /// Note: 現在の実装では、軸のサイズが定数（Expr::Const）の場合のみサポートされます。
    /// 変数サイズの軸に対してはパニックします。
    ///
    /// # パニック
    /// - axis が範囲外の場合
    /// - 軸のサイズが定数でない場合
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    /// use harp::graph::{Graph, DType};
    ///
    /// let mut graph = Graph::new();
    /// let x = Tensor::from_graph_node(
    ///     graph.input("x").with_dtype(DType::F32).with_shape([10, 20]).build(),
    ///     true
    /// );
    /// let mean_x = x.mean(1);  // 軸1の平均を計算
    /// ```
    pub fn mean(&self, axis: usize) -> Tensor {
        use harp::graph::shape::Expr;

        let shape = self.data.view.shape();
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
        &self.sum(axis) * (1.0f32 / size_value)
    }

    /// 分散を計算: var(x, axis)
    ///
    /// 不偏分散を計算します: E[(x - mean(x))^2]
    ///
    /// Note: この実装は効率的ではありません（meanを2回計算）。
    /// より効率的な実装には E[x^2] - E[x]^2 を使用できますが、
    /// 数値安定性の問題があります。
    ///
    /// # 例
    /// ```no_run
    /// use harp_autograd::Tensor;
    /// use harp::graph::{Graph, DType};
    ///
    /// let mut graph = Graph::new();
    /// let x = Tensor::from_graph_node(
    ///     graph.input("x").with_dtype(DType::F32).with_shape([3, 4]).build(),
    ///     true
    /// );
    /// let var = x.variance(1);  // 軸1の分散を計算
    /// ```
    pub fn variance(&self, axis: usize) -> Tensor {
        // 平均を計算
        let x_mean = self.mean(axis);

        // meanの次元を復元してbroadcast可能にする
        let x_mean_view = x_mean
            .data
            .view
            .clone()
            .unsqueeze(axis)
            .expand(self.data.view.shape().to_vec());
        let x_mean_expanded = Tensor::from_graph_node(x_mean.data.view(x_mean_view), false);

        // (x - mean)^2 の平均を計算
        (self - &x_mean_expanded).square().mean(axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harp::graph::{DType, Graph};

    #[test]
    fn test_log() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );

        let log_x = x.log();
        assert_eq!(log_x.data.view.ndim(), 1);
    }

    #[test]
    fn test_exp() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );

        let exp_x = x.exp();
        assert_eq!(exp_x.data.view.ndim(), 1);
    }

    #[test]
    fn test_cos() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );

        let cos_x = x.cos();
        assert_eq!(cos_x.data.view.ndim(), 1);
    }

    #[test]
    fn test_rsqrt() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );

        let rsqrt_x = x.rsqrt();
        assert_eq!(rsqrt_x.data.view.ndim(), 1);
    }

    #[test]
    fn test_square() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([5])
                .build(),
            true,
        );

        let squared = x.square();
        assert_eq!(squared.data.view.ndim(), 1);
    }

    #[test]
    fn test_powi() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([5])
                .build(),
            true,
        );

        let cubed = x.powi(3);
        assert_eq!(cubed.data.view.ndim(), 1);
    }

    #[test]
    #[should_panic(expected = "powi: n must be positive")]
    fn test_powi_zero() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([5])
                .build(),
            true,
        );

        let _ = x.powi(0);
    }

    #[test]
    fn test_abs_square() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );

        let abs_sq = x.abs_square();
        assert_eq!(abs_sq.data.view.ndim(), 1);
    }

    #[test]
    fn test_min() {
        let mut graph = Graph::new();
        let a = Tensor::from_graph_node(
            graph
                .input("a")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );
        let b = Tensor::from_graph_node(
            graph
                .input("b")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );

        let min_ab = a.min(&b);
        assert_eq!(min_ab.data.view.ndim(), 1);
    }

    #[test]
    fn test_clamp() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            true,
        );
        let min_val = Tensor::from_graph_node(
            graph
                .input("min")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            false,
        );
        let max_val = Tensor::from_graph_node(
            graph
                .input("max")
                .with_dtype(DType::F32)
                .with_shape([10])
                .build(),
            false,
        );

        let clamped = x.clamp(&min_val, &max_val);
        assert_eq!(clamped.data.view.ndim(), 1);
    }

    #[test]
    fn test_mean() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([10, 20])
                .build(),
            true,
        );

        let mean_x = x.mean(1);
        assert_eq!(mean_x.data.view.ndim(), 1);
    }

    #[test]
    fn test_variance() {
        let mut graph = Graph::new();
        let x = Tensor::from_graph_node(
            graph
                .input("x")
                .with_dtype(DType::F32)
                .with_shape([3, 4])
                .build(),
            true,
        );

        let var_x = x.variance(1);
        assert_eq!(var_x.data.view.ndim(), 1);
    }
}
