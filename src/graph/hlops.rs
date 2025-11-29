// Keep lines 1-316
//! 高レベル演算のヘルパー関数
//!
//! このモジュールは既存の基本的なグラフ演算を組み合わせて、
//! より高レベルな数学的演算や便利な演算を提供します。

use crate::graph::GraphNode;
use crate::graph::custom_builder;
use crate::graph::ops::{
    CumulativeOp, ElementwiseOp, GraphOp, ReduceOp, infer_dtype, infer_view, max, recip, reduce_sum,
};

impl GraphNode {
    /// 二乗: x^2
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![10, 20]);
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
    /// let x = graph.input("x", DType::F32, vec![10]);
    /// let min_val = graph.input("min", DType::F32, vec![10]);
    /// let max_val = graph.input("max", DType::F32, vec![10]);
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

    // ============================================================================
    // カスタム演算
    // ============================================================================

    /// カスタム関数演算を作成
    ///
    /// 完全なAstNode::Functionを直接指定してカスタム演算を定義します。
    /// 関数内ではプレースホルダー変数（input0, output, shape0, ridx0等）を使用します。
    ///
    /// # Example
    /// ```
    /// use harp::prelude::*;
    /// use harp::ast::{AstNode, DType as AstDType, FunctionKind, Scope, helper::*};
    /// use harp::graph::custom_placeholders as ph;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![10]);
    ///
    /// // x^2 を計算するカスタム関数
    /// let func = function(
    ///     None::<String>,
    ///     FunctionKind::Normal,
    ///     vec![],
    ///     AstDType::Tuple(vec![]),
    ///     range(
    ///         &ph::ridx(0),
    ///         const_int(0),
    ///         const_int(1),
    ///         var(&ph::shape(0)),
    ///         block(vec![
    ///             store(
    ///                 var(ph::OUTPUT),
    ///                 var(&ph::ridx(0)),
    ///                 load(var(&ph::input(0)), var(&ph::ridx(0)), AstDType::F32)
    ///                   * load(var(&ph::input(0)), var(&ph::ridx(0)), AstDType::F32),
    ///             ),
    ///         ], Scope::new()),
    ///     ),
    /// );
    /// let y = x.custom_function(func);
    /// ```
    pub fn custom_function(self, function: crate::ast::AstNode) -> Self {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode::new(dtype, GraphOp::Custom { ast: function }, vec![self], view)
    }

    /// カスタム関数演算を複数入力で作成
    pub fn custom_function_multi(inputs: Vec<Self>, function: crate::ast::AstNode) -> Self {
        if inputs.is_empty() {
            panic!("custom_function_multi requires at least one input");
        }

        let dtype = inputs[0].dtype.clone();
        let view = inputs[0].view.clone();

        GraphNode::new(dtype, GraphOp::Custom { ast: function }, inputs, view)
    }

    /// カスタムAST単項演算を作成
    ///
    /// 任意のASTノード（式）を使って要素ごとの演算を定義できます。
    /// AST内の`Wildcard("0")`が入力（self）に対応します。
    /// 内部的に完全な関数に変換されます。
    ///
    /// # Example
    /// ```
    /// use harp::prelude::*;
    /// use harp::ast::AstNode;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, [10]);
    ///
    /// // cos(x) をカスタム演算として表現
    /// let ast = AstNode::Call {
    ///     name: "cos".to_string(),
    ///     args: vec![AstNode::Wildcard("0".to_string())],
    /// };
    /// let y = x.custom_elementwise(ast);
    /// ```
    pub fn custom_elementwise(self, expr: crate::ast::AstNode) -> Self {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        let ndim = view.shape().len();

        let function = custom_builder::build_elementwise_function(ndim, 1, expr);

        GraphNode::new(dtype, GraphOp::Custom { ast: function }, vec![self], view)
    }

    /// カスタムAST二項演算を作成
    ///
    /// 任意のASTノードを使って2入力の要素ごとの演算を定義できます。
    /// AST内の`Wildcard("0")`が第1入力（self）、`Wildcard("1")`が第2入力（other）に対応します。
    ///
    /// # Example
    /// ```
    /// use harp::prelude::*;
    /// use harp::ast::AstNode;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, [10]);
    /// let y = graph.input("y", DType::F32, [10]);
    ///
    /// // pow(x, y) をカスタム演算として表現
    /// let ast = AstNode::Call {
    ///     name: "pow".to_string(),
    ///     args: vec![
    ///         AstNode::Wildcard("0".to_string()),
    ///         AstNode::Wildcard("1".to_string()),
    ///     ],
    /// };
    /// let z = x.custom_elementwise_binary(y, ast);
    /// ```
    pub fn custom_elementwise_binary(self, other: Self, expr: crate::ast::AstNode) -> Self {
        let dtype = infer_dtype(&self.dtype, &other.dtype);
        let view = infer_view(&self.view, &other.view);
        let ndim = view.shape().len();

        let function = custom_builder::build_elementwise_function(ndim, 2, expr);

        GraphNode::new(
            dtype,
            GraphOp::Custom { ast: function },
            vec![self, other],
            view,
        )
    }

    /// カスタムAST多入力演算を作成
    ///
    /// 任意の数の入力を持つカスタム演算を定義できます。
    /// AST内の`Wildcard("0")`, `Wildcard("1")`, ... が対応する入力に置換されます。
    ///
    /// # Example
    /// ```
    /// use harp::prelude::*;
    /// use harp::ast::AstNode;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a", DType::F32, [10]);
    /// let b = graph.input("b", DType::F32, [10]);
    /// let c = graph.input("c", DType::F32, [10]);
    ///
    /// // (a + b) * c をカスタム演算として表現
    /// let ast = AstNode::Mul(
    ///     Box::new(AstNode::Add(
    ///         Box::new(AstNode::Wildcard("0".to_string())),
    ///         Box::new(AstNode::Wildcard("1".to_string())),
    ///     )),
    ///     Box::new(AstNode::Wildcard("2".to_string())),
    /// );
    /// let result = GraphNode::custom_elementwise_multi(vec![a, b, c], ast);
    /// ```
    pub fn custom_elementwise_multi(inputs: Vec<Self>, expr: crate::ast::AstNode) -> Self {
        if inputs.is_empty() {
            panic!("custom_elementwise_multi requires at least one input");
        }

        let dtype = inputs[0].dtype.clone();
        let view = inputs[0].view.clone();
        let ndim = view.shape().len();
        let num_inputs = inputs.len();

        let function = custom_builder::build_elementwise_function(ndim, num_inputs, expr);

        GraphNode::new(dtype, GraphOp::Custom { ast: function }, inputs, view)
    }

    /// カスタムAST Reduce演算を作成
    ///
    /// Elementwise演算 → Reduce演算のパターンを単一ノードで表現します。
    /// AST内の`Wildcard("0")`, `Wildcard("1")`, ... が対応する入力に置換されます。
    ///
    /// # Example
    /// ```
    /// use harp::prelude::*;
    /// use harp::ast::helper::wildcard;
    /// use harp::graph::ReduceOp;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a", DType::F32, vec![10, 20]);
    /// let b = graph.input("b", DType::F32, vec![10, 20]);
    ///
    /// // reduce_sum(a * b, axis=1) をカスタム演算として表現
    /// let expr = wildcard("0") * wildcard("1");
    /// let result = GraphNode::custom_reduce(vec![a, b], expr, ReduceOp::Sum, 1);
    /// ```
    pub fn custom_reduce(
        inputs: Vec<Self>,
        expr: crate::ast::AstNode,
        reduce_op: ReduceOp,
        axis: usize,
    ) -> Self {
        if inputs.is_empty() {
            panic!("custom_reduce requires at least one input");
        }

        let dtype = inputs[0].dtype.clone();
        let input_view = inputs[0].view.clone();
        let ndim = input_view.shape().len();
        let num_inputs = inputs.len();

        // Reduce後のViewを計算（指定軸を削除）
        let mut new_shape = input_view.shape().to_vec();
        if axis < new_shape.len() {
            new_shape.remove(axis);
        }
        let view = crate::graph::shape::View::contiguous(new_shape);

        let function =
            custom_builder::build_reduce_function(ndim, num_inputs, axis, &reduce_op, expr);

        GraphNode::new(dtype, GraphOp::Custom { ast: function }, inputs, view)
    }

    /// カスタムAST Cumulative演算を作成
    ///
    /// Elementwise演算 → Cumulative演算のパターンを単一ノードで表現します。
    /// AST内の`Wildcard("0")`, `Wildcard("1")`, ... が対応する入力に置換されます。
    ///
    /// # Example
    /// ```
    /// use harp::prelude::*;
    /// use harp::ast::helper::wildcard;
    /// use harp::graph::CumulativeOp;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![10, 20]);
    ///
    /// // cumsum(x * x, axis=1) をカスタム演算として表現
    /// let expr = wildcard("0") * wildcard("0");
    /// let result = GraphNode::custom_cumulative(vec![x], expr, CumulativeOp::Sum, 1);
    /// ```
    pub fn custom_cumulative(
        inputs: Vec<Self>,
        expr: crate::ast::AstNode,
        cumulative_op: CumulativeOp,
        axis: usize,
    ) -> Self {
        if inputs.is_empty() {
            panic!("custom_cumulative requires at least one input");
        }

        let dtype = inputs[0].dtype.clone();
        let view = inputs[0].view.clone();
        let ndim = view.shape().len();
        let num_inputs = inputs.len();

        let function =
            custom_builder::build_cumulative_function(ndim, num_inputs, axis, &cumulative_op, expr);

        GraphNode::new(dtype, GraphOp::Custom { ast: function }, inputs, view)
    }

    // ============================================================================
    // 乱数初期化
    // ============================================================================

    /// 指定した形状の標準正規分布（平均0、標準偏差1）の乱数テンソルを生成
    ///
    /// Box-Muller法を使用して一様乱数から正規乱数を生成します。
    ///
    /// # アルゴリズム
    /// 2つの独立した一様乱数 U1, U2 ∈ (0, 1) から：
    /// - Z = sqrt(-2 * ln(U1)) * cos(2 * π * U2)
    ///
    /// Z は標準正規分布に従います。
    ///
    /// # 引数
    /// - `shape`: テンソルの形状（静的な`usize`/`isize`または動的な`Expr`を受け付ける）
    ///
    /// # 例
    /// ```ignore
    /// use harp::prelude::*;
    ///
    /// // 静的な形状: 2x3の正規乱数テンソル
    /// let randn_node = GraphNode::randn(vec![2, 3]);
    ///
    /// // 動的な形状
    /// let batch_size = shape::Expr::var("batch");
    /// let randn_node = GraphNode::randn(vec![batch_size, 64.into()]);
    /// ```
    pub fn randn<E: Into<crate::graph::shape::Expr> + Clone, I: IntoIterator<Item = E>>(
        shape: I,
    ) -> Self {
        use crate::graph::shape::Expr;
        use std::f32::consts::PI;

        let shape_exprs: Vec<Expr> = shape.into_iter().map(|e| e.into()).collect();

        // 2つの独立した一様乱数テンソルを生成
        let u1 = GraphNode::rand(shape_exprs.clone());
        let u2 = GraphNode::rand(shape_exprs);

        // Box-Muller変換:
        // Z = sqrt(-2 * ln(U1)) * cos(2 * π * U2)
        //
        // Note: U1が0に非常に近い場合、ln(U1)は負の大きな値になりますが、
        // 実用上は問題になることは稀です。

        // sqrt(-2 * ln(U1))
        // -2 * ln(U1) = -2 * log2(U1) / log2(e) = log2(U1) * (-2 / log2(e))
        const NEG_2_DIV_LOG2_E: f32 = -2.0 / std::f32::consts::LOG2_E;
        let r = (u1.log2() * NEG_2_DIV_LOG2_E).sqrt();

        // cos(2 * π * U2)
        let theta = u2 * (2.0 * PI);
        let cos_theta = theta.cos();

        // Z = r * cos(theta)
        r * cos_theta
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
        let x = graph.input("x", DType::F32, vec![10, 20]);

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
        let x = graph.input("x", DType::F32, vec![5]);

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
        let x = graph.input("x", DType::F32, vec![5]);

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
        let x = graph.input("x", DType::F32, vec![5]);

        let _ = x.powi(0);
    }

    #[test]
    fn test_abs_square() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, vec![10]);

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
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![10]);
        let min_val = graph.input("min", DType::F32, vec![10]);
        let max_val = graph.input("max", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![10, 20, 30]);

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
        let x = graph.input("x", DType::F32, vec![10, 20]);

        let _ = x.mean(3);
    }

    #[test]
    #[should_panic(expected = "matmul: not yet implemented")]
    fn test_matmul_not_implemented() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![20, 30]);

        let _ = a.matmul(b);
    }

    #[test]
    #[should_panic(expected = "matmul: first argument must be 2D")]
    fn test_matmul_wrong_dim_a() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10, 20]);

        let _ = a.matmul(b);
    }

    #[test]
    #[should_panic(expected = "batch_matmul: not yet implemented")]
    fn test_batch_matmul_not_implemented() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![5, 10, 20]);
        let b = graph.input("b", DType::F32, vec![5, 20, 30]);

        let _ = a.batch_matmul(b);
    }

    #[test]
    fn test_log2() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![5, 5]);

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
        let x = graph.input("x", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![10]);

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
        let x = graph.input("x", DType::F32, vec![10]);

        let result = x.rsqrt();

        // rsqrt(x) = recip(sqrt(x))
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
    }

    // ============================================================================
    // 乱数初期化テスト
    // ============================================================================

    #[test]
    fn test_randn() {
        // 静的な形状で正規乱数テンソルを生成
        let result = GraphNode::randn(vec![10, 20]);

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 2);
        assert_eq!(result.view.shape()[0], Expr::from(10));
        assert_eq!(result.view.shape()[1], Expr::from(20));
    }

    #[test]
    fn test_randn_dynamic_shape() {
        // 動的な形状で正規乱数テンソルを生成
        let batch = Expr::Var("batch".to_string());
        let result = GraphNode::randn(vec![batch.clone(), Expr::from(64)]);

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 2);
        assert_eq!(result.view.shape()[0], batch);
        assert_eq!(result.view.shape()[1], Expr::from(64));
    }

    #[test]
    fn test_randn_1d() {
        // 1次元の正規乱数テンソルを生成
        let result = GraphNode::randn(vec![100]);

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(result.view.ndim(), 1);
        assert_eq!(result.view.shape()[0], Expr::from(100));
    }
}
