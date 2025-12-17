// GraphNodeのView関連の操作を提供するモジュール

use crate::graph::GraphNode;
use crate::graph::conv::{ConvParams, IntoSpatialParams};
use crate::graph::ops::GraphOp;
use crate::graph::shape::{Expr, View};

impl GraphNode {
    /// Viewを変更した新しいノードを作成
    ///
    /// このメソッドは、既存のノードに対してView操作（permute, unsqueeze, expand等）を
    /// 適用した新しいノードを作成します。
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a", DType::F32, vec![3, 4]);
    ///
    /// // Viewを変更（転置）
    /// let transposed_view = a.view.clone().permute(vec![1, 0]);
    /// let a_transposed = a.view(transposed_view);
    /// ```
    pub fn view(&self, new_view: View) -> Self {
        Self::new(
            self.dtype.clone(),
            GraphOp::View(new_view.clone()),
            vec![self.clone()],
            new_view,
        )
    }

    /// Viewを連続レイアウトに変換（contiguous化）
    ///
    /// 非線形View（IndexExpr）やストライドが不連続なViewを、
    /// メモリ上で連続したレイアウトに変換します。
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a", DType::F32, vec![3, 4]);
    ///
    /// // 転置後にcontiguous化
    /// let transposed = a.view(a.view.clone().permute(vec![1, 0]));
    /// let contiguous = transposed.contiguous();
    /// ```
    pub fn contiguous(&self) -> Self {
        let new_view = View::contiguous(self.view.shape().to_vec());
        Self::new(
            self.dtype.clone(),
            GraphOp::Contiguous {},
            vec![self.clone()],
            new_view,
        )
    }

    /// 必要な場合のみcontiguous化
    ///
    /// 既にLinear Viewの場合はそのまま返し、IndexExprの場合のみcontiguous化します。
    fn ensure_linear(&self) -> Self {
        if self.view.is_linear() {
            self.clone()
        } else {
            self.contiguous()
        }
    }

    /// テンソルの形状を変更（reshape）
    ///
    /// 要素数が同じ場合に使用可能です。
    /// IndexExpr Viewの場合は自動的にcontiguous化してからreshapeします。
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    /// use harp::graph::shape::Expr;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a", DType::F32, vec![3, 4]);
    ///
    /// // (3, 4) -> (12,) にreshape
    /// let flattened = a.reshape(vec![Expr::from(12)]);
    ///
    /// // (3, 4) -> (2, 6) にreshape
    /// let reshaped = a.reshape(vec![Expr::from(2), Expr::from(6)]);
    /// ```
    pub fn reshape(&self, new_shape: Vec<Expr>) -> Self {
        // IndexExprの場合は自動でcontiguous化
        let src = self.ensure_linear();
        let new_view = src.view.clone().reshape(new_shape);
        src.view(new_view)
    }

    /// ブロードキャストのためにshapeを拡張
    ///
    /// 指定した軸を繰り返す（サイズ1の軸のみ対応）
    ///
    /// # Arguments
    /// * `axis` - 繰り返す軸のインデックス
    /// * `times` - 繰り返し回数（結果のサイズ）
    ///
    /// # パニック
    /// - 軸が範囲外の場合
    /// - 指定軸のサイズが1でない場合
    pub fn repeat(&self, axis: usize, times: impl Into<Expr>) -> Self {
        let new_view = self.view.clone().repeat(axis, times);
        self.view(new_view)
    }

    /// 複数軸をブロードキャストして目標形状に拡張
    ///
    /// サイズ1の軸を自動検出し、目標形状に合わせて `repeat` をチェーンします。
    /// サイズ1でない軸は変更されません（目標と一致する必要があります）。
    ///
    /// # Arguments
    /// * `target_shape` - 目標形状
    ///
    /// # パニック
    /// - target_shapeの次元数が現在の次元数と一致しない場合
    /// - サイズ1でない軸が目標形状と一致しない場合
    pub fn broadcast_to(&self, target_shape: Vec<Expr>) -> Self {
        assert_eq!(
            target_shape.len(),
            self.view.ndim(),
            "broadcast_to: target shape must have same ndim"
        );

        let mut result = self.clone();
        let current_shape = result.view.shape().to_vec();

        for (axis, (current, target)) in current_shape.iter().zip(target_shape.iter()).enumerate() {
            if current.is_one() && !target.is_one() {
                result = result.repeat(axis, target.clone());
            }
            // 既にtargetと一致しているか、サイズ1でない場合は何もしない
        }

        result
    }

    /// unfold操作（スライディングウィンドウ）
    ///
    /// 畳み込みの前処理として、入力から重複するパッチを抽出します。
    /// 次元数はパラメータから自動判定されます。
    ///
    /// # 引数
    /// - `kernel_size`: ウィンドウサイズ
    /// - `stride`: ストライド（スライディングウィンドウの移動距離）
    /// - `dilation`: 膨張率（カーネル要素間の距離）
    /// - `groups`: グループ数（グループ畳み込み用、通常は1）
    ///
    /// # 入出力形状
    /// - groups=1の場合:
    ///   - N次元入力（空間のみ）: (...) -> (k1, k2, ..., L1', L2', ...)
    ///   - (N+1)次元入力（チャネル付き）: (C, ...) -> (C, k1, k2, ..., L1', L2', ...)
    /// - groups>1の場合:
    ///   - (C, ...) -> (groups, C/groups, k1, k2, ..., L1', L2', ...)
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    ///
    /// // 1D unfold: (10,) -> (3, 8)
    /// let x1 = graph.input("x1", DType::F32, vec![10]);
    /// let unfolded1 = x1.unfold(3, 1, 1, 1);
    ///
    /// // 2D unfold: (3, 32, 32) -> (3, 3, 3, 30, 30)
    /// let x2 = graph.input("x2", DType::F32, vec![3, 32, 32]);
    /// let unfolded2 = x2.unfold((3, 3), (1, 1), (1, 1), 1);
    ///
    /// // 3D unfold: (2, 16, 16, 16) -> (2, 3, 3, 3, 14, 14, 14)
    /// let x3 = graph.input("x3", DType::F32, vec![2, 16, 16, 16]);
    /// let unfolded3 = x3.unfold((3, 3, 3), (1, 1, 1), (1, 1, 1), 1);
    ///
    /// // グループ畳み込み: (6, 16, 16) -> (2, 3, 3, 3, 14, 14)
    /// let x4 = graph.input("x4", DType::F32, vec![6, 16, 16]);
    /// let unfolded4 = x4.unfold((3, 3), (1, 1), (1, 1), 2);
    /// ```
    pub fn unfold<S: IntoSpatialParams>(
        &self,
        kernel_size: S,
        stride: S,
        dilation: S,
        groups: usize,
    ) -> Self {
        let kernel_size_vec = kernel_size.into_vec();
        let stride_vec = stride.into_vec();
        let dilation_vec = dilation.into_vec();

        let spatial_dims = kernel_size_vec.len();

        // パラメータの次元数を検証
        assert_eq!(
            stride_vec.len(),
            spatial_dims,
            "stride must have {} elements",
            spatial_dims
        );
        assert_eq!(
            dilation_vec.len(),
            spatial_dims,
            "dilation must have {} elements",
            spatial_dims
        );

        let params = ConvParams::new(kernel_size_vec, stride_vec, dilation_vec, groups);
        self.unfold_nd(&params)
    }

    /// N次元unfold操作（内部API）
    ///
    /// ConvParamsを使用した低レベルunfold操作。
    /// conv_ndなど内部実装で使用。
    /// IndexExpr Viewの場合は自動的にcontiguous化してからunfoldします。
    pub(crate) fn unfold_nd(&self, params: &ConvParams) -> Self {
        // IndexExprの場合は自動でcontiguous化
        let src = self.ensure_linear();
        let new_view = src.view.clone().unfold_nd(params);
        src.view(new_view)
    }

    /// テンソルをパディング
    ///
    /// 各軸に対して前後にパディングを追加します。
    ///
    /// # 引数
    /// - `padding`: 各軸の(前, 後)パディング量のベクタ
    /// - `value`: パディング値（通常0.0）
    ///
    /// # パニック
    /// - padding.len()がテンソルのndimと一致しない場合
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![3, 32, 32]);
    ///
    /// // 2D画像の高さ・幅に1ピクセルずつパディング
    /// // (3, 32, 32) -> (3, 34, 34)
    /// let padded = x.pad(vec![(0, 0), (1, 1), (1, 1)], 0.0);
    /// ```
    pub fn pad(&self, padding: Vec<(usize, usize)>, value: f32) -> Self {
        assert_eq!(
            padding.len(),
            self.view.ndim(),
            "padding length must match tensor ndim"
        );

        // パディング後のshapeを計算
        let old_shape = self.view.shape();
        let new_shape: Vec<Expr> = old_shape
            .iter()
            .zip(padding.iter())
            .map(|(size, (before, after))| {
                size.clone() + Expr::from(*before as isize) + Expr::from(*after as isize)
            })
            .collect();

        let new_view = View::contiguous(new_shape);
        Self::new(
            self.dtype.clone(),
            GraphOp::Pad { padding, value },
            vec![self.clone()],
            new_view,
        )
    }

    /// テンソルの一部を切り出し
    ///
    /// 各軸に対してstart:endの範囲でスライスします（endは含まない）。
    ///
    /// # 引数
    /// - `ranges`: 各軸の(start, end)範囲のベクタ
    ///
    /// # パニック
    /// - ranges.len()がテンソルのndimと一致しない場合
    /// - start >= endの場合
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![10, 20]);
    ///
    /// // [10, 20] から [2:5, 3:18] を切り出し -> [3, 15]
    /// let sliced = x.slice(vec![(2, 5), (3, 18)]);
    /// ```
    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Self {
        assert_eq!(
            ranges.len(),
            self.view.ndim(),
            "ranges length must match tensor ndim"
        );

        // 範囲のバリデーション
        for (i, (start, end)) in ranges.iter().enumerate() {
            assert!(
                start < end,
                "Invalid range for axis {}: start ({}) must be less than end ({})",
                i,
                start,
                end
            );
        }

        // スライス後のshapeを計算
        let new_shape: Vec<Expr> = ranges
            .iter()
            .map(|(start, end)| Expr::from((end - start) as isize))
            .collect();

        let new_view = View::contiguous(new_shape);
        Self::new(
            self.dtype.clone(),
            GraphOp::Slice { ranges },
            vec![self.clone()],
            new_view,
        )
    }

    /// fold操作（unfoldの逆操作、col2im）
    ///
    /// unfoldされたテンソルを元の形状に戻します。
    /// 重複する部分は加算されます。
    /// 次元数はパラメータから自動判定されます。
    ///
    /// # 引数
    /// - `output_size`: 出力サイズ（unfold前の空間サイズ）
    /// - `kernel_size`: ウィンドウサイズ
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数（グループ畳み込み用、通常は1）
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    ///
    /// // 1D fold: (2, 3, 8) -> (2, 10)
    /// let x1 = graph.input("x1", DType::F32, vec![2, 3, 8]);
    /// let folded1 = x1.fold(vec![2, 10], 3, 1, 1, 1);
    ///
    /// // 2D fold: (3, 3, 3, 30, 30) -> (3, 32, 32)
    /// let x2 = graph.input("x2", DType::F32, vec![3, 3, 3, 30, 30]);
    /// let folded2 = x2.fold(vec![3, 32, 32], (3, 3), (1, 1), (1, 1), 1);
    ///
    /// // 3D fold: (2, 3, 3, 3, 14, 14, 14) -> (2, 16, 16, 16)
    /// let x3 = graph.input("x3", DType::F32, vec![2, 3, 3, 3, 14, 14, 14]);
    /// let folded3 = x3.fold(vec![2, 16, 16, 16], (3, 3, 3), (1, 1, 1), (1, 1, 1), 1);
    /// ```
    pub fn fold<S: IntoSpatialParams>(
        &self,
        output_size: Vec<usize>,
        kernel_size: S,
        stride: S,
        dilation: S,
        groups: usize,
    ) -> Self {
        let kernel_size_vec = kernel_size.into_vec();
        let stride_vec = stride.into_vec();
        let dilation_vec = dilation.into_vec();

        let spatial_dims = kernel_size_vec.len();

        // パラメータの次元数を検証
        assert_eq!(
            stride_vec.len(),
            spatial_dims,
            "stride must have {} elements",
            spatial_dims
        );
        assert_eq!(
            dilation_vec.len(),
            spatial_dims,
            "dilation must have {} elements",
            spatial_dims
        );

        let params = ConvParams::new(kernel_size_vec, stride_vec, dilation_vec, groups);
        self.fold_nd(output_size, &params)
    }

    /// N次元fold操作（内部API）
    ///
    /// conv_transpose_ndなど内部実装で使用。
    pub(crate) fn fold_nd(&self, output_size: Vec<usize>, params: &ConvParams) -> Self {
        let new_shape: Vec<Expr> = output_size
            .iter()
            .map(|&s| Expr::from(s as isize))
            .collect();
        let new_view = View::contiguous(new_shape);

        Self::new(
            self.dtype.clone(),
            GraphOp::Fold {
                output_size,
                kernel_size: params.kernel_size.clone(),
                stride: params.stride.clone(),
                dilation: params.dilation.clone(),
                groups: params.groups,
            },
            vec![self.clone()],
            new_view,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_contiguous() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![3, 4]);
        let contiguous = a.contiguous();

        assert!(matches!(contiguous.op, GraphOp::Contiguous {}));
        assert_eq!(contiguous.view.shape(), &[Expr::from(3), Expr::from(4)]);
        assert!(contiguous.view.is_linear());
    }

    #[test]
    fn test_reshape_with_index_expr_auto_contiguous() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![3, 4]);

        // tile操作でIndexExprに変換
        let tiled = a.view(a.view.clone().tile(0, 2)); // shape: [6, 4]
        assert!(!tiled.view.is_linear());

        // reshapeは自動でcontiguous化される（panicしない）
        let reshaped = tiled.reshape(vec![Expr::from(24)]);
        assert_eq!(reshaped.view.shape(), &[Expr::from(24)]);
        assert!(reshaped.view.is_linear());

        // 入力ノードを辿るとContiguousノードがある
        assert!(matches!(reshaped.src[0].op, GraphOp::Contiguous {}));
    }

    #[test]
    fn test_unfold_with_index_expr_auto_contiguous() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);

        // tile操作でIndexExprに変換
        let tiled = a.view(a.view.clone().tile(0, 2)); // shape: [20]
        assert!(!tiled.view.is_linear());

        // unfoldは自動でcontiguous化される（panicしない）
        let unfolded = tiled.unfold(3, 1, 1, 1);

        // 入力ノードを辿るとContiguousノードがある
        // unfolded -> View -> tiled_contiguous -> Contiguous -> tiled -> View -> a
        let view_src = &unfolded.src[0];
        assert!(matches!(view_src.op, GraphOp::Contiguous {}));
    }

    #[test]
    fn test_reshape_linear_no_contiguous() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![3, 4]);

        // LinearのままreshapeするとContiguousノードは挿入されない
        let reshaped = a.reshape(vec![Expr::from(12)]);
        // reshaped -> a (Inputノード) であり、Contiguousは挿入されない
        assert!(!matches!(reshaped.src[0].op, GraphOp::Contiguous {}));
    }

    #[test]
    fn test_ensure_linear_idempotent() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![3, 4]);

        // 既にLinearなノードにensure_linearを適用しても変わらない
        let ensured = a.ensure_linear();
        assert_eq!(a.as_ptr(), ensured.as_ptr());
    }
}
