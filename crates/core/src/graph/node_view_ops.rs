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
    /// use harp_core::prelude::*;
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
    /// use harp_core::prelude::*;
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
    /// use harp_core::prelude::*;
    /// use harp_core::graph::shape::Expr;
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
    /// use harp_core::prelude::*;
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
    /// 動的shapeに対応しており、パディング量にExpr（シンボリック式）を使用可能です。
    ///
    /// # 引数
    /// - `padding`: 各軸の(前, 後)パディング量のベクタ（usize, isize, Expr等が使用可能）
    /// - `value`: パディング値（通常0.0）
    ///
    /// # パニック
    /// - padding.len()がテンソルのndimと一致しない場合
    /// - 静的に評価可能なパディング量が負の場合
    ///
    /// # 例
    /// ```no_run
    /// use harp_core::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![3, 32, 32]);
    ///
    /// // 2D画像の高さ・幅に1ピクセルずつパディング（静的）
    /// // (3, 32, 32) -> (3, 34, 34)
    /// let padded = x.pad(vec![(0, 0), (1, 1), (1, 1)], 0.0);
    ///
    /// // 動的パディング（Exprを使用）
    /// // let n = Expr::Var("N".to_string());
    /// // let padded = x.pad(vec![(n.clone(), n)], 0.0);
    /// ```
    pub fn pad<E: Into<Expr>>(&self, padding: Vec<(E, E)>, value: f32) -> Self {
        // Exprに変換
        let padding: Vec<(Expr, Expr)> = padding
            .into_iter()
            .map(|(before, after)| (before.into(), after.into()))
            .collect();

        assert_eq!(
            padding.len(),
            self.view.ndim(),
            "padding length must match tensor ndim"
        );

        // 静的にチェック可能な場合は負値チェック
        for (i, (before, after)) in padding.iter().enumerate() {
            if let Some(v) = before.as_const() {
                assert!(
                    v >= 0,
                    "padding[{}].before must be non-negative, got {}",
                    i,
                    v
                );
            }
            if let Some(v) = after.as_const() {
                assert!(
                    v >= 0,
                    "padding[{}].after must be non-negative, got {}",
                    i,
                    v
                );
            }
        }

        // パディング後のshapeを計算
        let old_shape = self.view.shape();
        let new_shape: Vec<Expr> = old_shape
            .iter()
            .zip(padding.iter())
            .map(|(size, (before, after))| size.clone() + before.clone() + after.clone())
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
    /// use harp_core::prelude::*;
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
            .map(|(start, end)| Expr::from((end - start) as i64))
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
    /// use harp_core::prelude::*;
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
            .map(|&s| Expr::from(s as i64))
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

    /// Gather操作（PyTorchのtorch.gatherに相当）
    ///
    /// 指定した軸に沿って、indexテンソルの値に従ってinputテンソルから要素を収集します。
    ///
    /// # 動作
    /// ```text
    /// output[i][j][k] = self[i][index[i][j][k]][k]  // dim=1の場合
    /// ```
    ///
    /// # 引数
    /// - `dim`: Gather軸（この軸のインデックスがindexテンソルの値で置き換わる）
    /// - `index`: インデックステンソル（整数型、outputと同じ形状）
    ///
    /// # 制約
    /// - indexテンソルの次元数はselfと同じである必要がある
    /// - dim以外の軸では、indexのサイズがselfのサイズ以下である必要がある
    /// - indexテンソルの値は[0, self.shape[dim])の範囲内である必要がある
    ///
    /// # 例
    /// ```no_run
    /// use harp_core::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let input = graph.input("input", DType::F32, vec![3, 4, 5]);
    /// let index = graph.input("index", DType::I32, vec![3, 2, 5]);
    ///
    /// // dim=1: output[i][j][k] = input[i][index[i][j][k]][k]
    /// let gathered = input.gather(1, &index);
    /// // output shape: [3, 2, 5]
    /// ```
    pub fn gather(&self, dim: usize, index: &GraphNode) -> Self {
        let input_shape = self.view.shape();
        let index_shape = index.view.shape();
        let ndim = input_shape.len();

        assert_eq!(
            ndim,
            index_shape.len(),
            "gather: input and index must have same number of dimensions"
        );
        assert!(
            dim < ndim,
            "gather: dim ({}) must be less than ndim ({})",
            dim,
            ndim
        );

        // 出力形状はindexと同じ
        let output_shape = index_shape.to_vec();

        // indexテンソルへのアクセスオフセットを構築（連続アクセス）
        // offset = Idx(0) * stride0 + Idx(1) * stride1 + ...
        let index_offset = build_contiguous_offset_with_shape(index_shape);

        // inputテンソルへのアクセスオフセットを構築
        // dim軸以外は通常のIdxを使用し、dim軸はLoadIndexでindexの値を使用
        let input_offset = build_gather_offset(input_shape, dim, index_offset);

        let new_view = View::IndexExpr {
            shape: output_shape,
            index_expr: input_offset.simplify(),
        };

        // srcに[input, index]を持たせる
        Self::new(
            self.dtype.clone(),
            GraphOp::View(new_view.clone()),
            vec![self.clone(), index.clone()],
            new_view,
        )
    }

    /// View連鎖をフラット化する
    ///
    /// View→View→...の連鎖を再帰的に辿り、最終的なViewと全てのsrcノードを
    /// フラットな配列として返します。LoadIndexのsrc_indexは適切にシフトされます。
    ///
    /// # Returns
    /// (flattened_view, flattened_src) のタプル
    /// - flattened_view: 合成されたView
    /// - flattened_src: マージされたsrc配列
    ///
    /// # Example
    /// ```
    /// use harp_core::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let input = graph.input("input", DType::F32, vec![3, 4, 5]);
    /// let index = graph.input("index", DType::I32, vec![3, 2, 5]);
    ///
    /// // gather → permute の連鎖
    /// let gathered = input.gather(1, &index);
    /// let permuted = gathered.view(gathered.view.clone().permute(vec![1, 0, 2]));
    ///
    /// // フラット化
    /// let (view, srcs) = permuted.flatten_view_chain();
    /// // srcs = [input, index] (Viewノードを除いた元のソース)
    /// ```
    pub fn flatten_view_chain(&self) -> (View, Vec<GraphNode>) {
        self.flatten_view_chain_internal(0)
    }

    /// View連鎖フラット化の内部実装
    ///
    /// # Arguments
    /// * `src_index_offset` - LoadIndexのsrc_indexに加算するオフセット
    fn flatten_view_chain_internal(&self, src_index_offset: isize) -> (View, Vec<GraphNode>) {
        match &self.op {
            GraphOp::View(_) => {
                // Viewノードの場合、src[0]を再帰的に処理
                if self.src.is_empty() {
                    // srcがない場合（異常ケース）
                    return (self.view.clone(), vec![]);
                }

                let primary_src = &self.src[0];

                // src[0]がViewノードの場合は再帰
                if matches!(primary_src.op, GraphOp::View(_)) {
                    // 現在のノードの追加src（src[1..]）を収集
                    let current_extra_srcs: Vec<GraphNode> = self.src[1..].to_vec();
                    let extra_count = current_extra_srcs.len() as isize;

                    // 再帰的にinner srcをフラット化
                    // innerのLoadIndexはさらにextra_count分シフトが必要
                    let (inner_view, inner_srcs) =
                        primary_src.flatten_view_chain_internal(src_index_offset + extra_count);

                    // Viewを合成
                    let outer_view = self.view.shift_load_index(src_index_offset);
                    let composed_view = View::compose(&outer_view, &inner_view);

                    // srcをマージ: [inner_srcs[0], current_extra_srcs..., inner_srcs[1..]...]
                    let mut merged_srcs = Vec::new();
                    if !inner_srcs.is_empty() {
                        merged_srcs.push(inner_srcs[0].clone());
                    }
                    merged_srcs.extend(current_extra_srcs);
                    if inner_srcs.len() > 1 {
                        merged_srcs.extend(inner_srcs[1..].iter().cloned());
                    }

                    (composed_view, merged_srcs)
                } else {
                    // src[0]がViewノードでない場合は終端
                    let shifted_view = self.view.shift_load_index(src_index_offset);
                    (shifted_view, self.src.clone())
                }
            }
            _ => {
                // Viewノードでない場合はそのまま返す
                (self.view.clone(), vec![self.clone()])
            }
        }
    }
}

/// 連続アクセス用のオフセット式を構築
/// offset = Idx(0) * stride(0) + Idx(1) * stride(1) + ... + Idx(n-1)
/// ここでstride(i) = shape[i+1] * shape[i+2] * ... * shape[n-1]
fn build_contiguous_offset_with_shape(shape: &[Expr]) -> Expr {
    let ndim = shape.len();
    if ndim == 0 {
        return Expr::Const(0);
    }

    // ストライドを計算
    // stride[i] = shape[i+1] * ... * shape[n-1]
    let mut strides = vec![Expr::Const(1); ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = (strides[i + 1].clone() * shape[i + 1].clone()).simplify();
    }

    // オフセット式を構築
    let mut expr = Expr::Const(0);
    for (i, stride) in strides.iter().enumerate() {
        expr = (expr + Expr::Idx(i) * stride.clone()).simplify();
    }
    expr
}

/// Gather操作用のinputアクセスオフセットを構築
fn build_gather_offset(input_shape: &[Expr], dim: usize, index_offset: Expr) -> Expr {
    let ndim = input_shape.len();
    if ndim == 0 {
        return Expr::Const(0);
    }

    // input用のストライドを計算
    // stride[i] = input_shape[i+1] * ... * input_shape[n-1]
    let mut strides = vec![Expr::Const(1); ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = (strides[i + 1].clone() * input_shape[i + 1].clone()).simplify();
    }

    // オフセット式を構築
    // dim軸以外: Idx(i) * stride[i]
    // dim軸: LoadIndex(1, index_offset) * stride[dim]
    let mut expr = Expr::Const(0);
    for (i, stride) in strides.iter().enumerate() {
        let term = if i == dim {
            // インデックステンソル（src[1]）から値を読み込む
            Expr::LoadIndex {
                src_index: 1,
                offset_expr: Box::new(index_offset.clone()),
            } * stride.clone()
        } else {
            Expr::Idx(i) * stride.clone()
        };
        expr = (expr + term).simplify();
    }
    expr
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

        assert!(matches!(contiguous.op, GraphOp::Contiguous));
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
        assert!(matches!(reshaped.src[0].op, GraphOp::Contiguous));
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
        assert!(matches!(view_src.op, GraphOp::Contiguous));
    }

    #[test]
    fn test_reshape_linear_no_contiguous() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![3, 4]);

        // LinearのままreshapeするとContiguousノードは挿入されない
        let reshaped = a.reshape(vec![Expr::from(12)]);
        // reshaped -> a (Inputノード) であり、Contiguousは挿入されない
        assert!(!matches!(reshaped.src[0].op, GraphOp::Contiguous));
    }

    #[test]
    fn test_ensure_linear_idempotent() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![3, 4]);

        // 既にLinearなノードにensure_linearを適用しても変わらない
        let ensured = a.ensure_linear();
        assert_eq!(a.as_ptr(), ensured.as_ptr());
    }

    #[test]
    fn test_gather_basic() {
        let mut graph = Graph::new();
        let input = graph.input("input", DType::F32, vec![3, 4, 5]);
        let index = graph.input("index", DType::I32, vec![3, 2, 5]);

        // dim=1でgather: output[i][j][k] = input[i][index[i][j][k]][k]
        let gathered = input.gather(1, &index);

        // 出力形状はindexと同じ
        assert_eq!(
            gathered.view.shape(),
            &[Expr::from(3), Expr::from(2), Expr::from(5)]
        );

        // ViewはIndexExpr
        assert!(!gathered.view.is_linear());

        // srcは[input, index]
        assert_eq!(gathered.src.len(), 2);
    }

    #[test]
    fn test_gather_dim0() {
        let mut graph = Graph::new();
        let input = graph.input("input", DType::F32, vec![4, 3]);
        let index = graph.input("index", DType::I32, vec![2, 3]);

        // dim=0でgather: output[i][j] = input[index[i][j]][j]
        let gathered = input.gather(0, &index);

        assert_eq!(gathered.view.shape(), &[Expr::from(2), Expr::from(3)]);
    }

    #[test]
    fn test_gather_1d() {
        let mut graph = Graph::new();
        let input = graph.input("input", DType::F32, vec![10]);
        let index = graph.input("index", DType::I32, vec![5]);

        // 1次元の場合: output[i] = input[index[i]]
        let gathered = input.gather(0, &index);

        assert_eq!(gathered.view.shape(), &[Expr::from(5)]);

        // index_exprにLoadIndexが含まれているはず
        if let View::IndexExpr { index_expr, .. } = &gathered.view {
            assert!(index_expr.contains_load_index());
        } else {
            panic!("Expected IndexExpr view");
        }
    }

    #[test]
    #[should_panic(expected = "gather: input and index must have same number of dimensions")]
    fn test_gather_dimension_mismatch() {
        let mut graph = Graph::new();
        let input = graph.input("input", DType::F32, vec![3, 4]);
        let index = graph.input("index", DType::I32, vec![3, 4, 5]); // 次元数が異なる

        let _ = input.gather(1, &index);
    }

    #[test]
    #[should_panic(expected = "gather: dim (2) must be less than ndim (2)")]
    fn test_gather_invalid_dim() {
        let mut graph = Graph::new();
        let input = graph.input("input", DType::F32, vec![3, 4]);
        let index = graph.input("index", DType::I32, vec![3, 4]);

        let _ = input.gather(2, &index); // dim=2は範囲外
    }

    #[test]
    fn test_gather_consecutive() {
        let mut graph = Graph::new();
        // input: [2, 3, 4]
        let input = graph.input("input", DType::F32, vec![2, 3, 4]);
        // index1: [2, 2, 4] - dim=1でgather
        let index1 = graph.input("index1", DType::I32, vec![2, 2, 4]);
        // index2: [2, 2, 2] - dim=2でgather
        let index2 = graph.input("index2", DType::I32, vec![2, 2, 2]);

        // 1回目のgather: output1[i][j][k] = input[i][index1[i][j][k]][k]
        let gather1 = input.gather(1, &index1);
        assert_eq!(
            gather1.view.shape(),
            &[Expr::from(2), Expr::from(2), Expr::from(4)]
        );

        // 2回目のgather: output2[i][j][k] = gather1[i][j][index2[i][j][k]]
        let gather2 = gather1.gather(2, &index2);
        assert_eq!(
            gather2.view.shape(),
            &[Expr::from(2), Expr::from(2), Expr::from(2)]
        );

        // 構造確認
        // gather2.src = [gather1, index2]
        assert_eq!(gather2.src.len(), 2);
        // gather2.src[0] (=gather1) は src = [input, index1]
        assert_eq!(gather2.src[0].src.len(), 2);

        // 両方ともIndexExpr
        assert!(!gather1.view.is_linear());
        assert!(!gather2.view.is_linear());

        // ViewにLoadIndexが含まれている
        if let View::IndexExpr { index_expr, .. } = &gather2.view {
            assert!(index_expr.contains_load_index());
        } else {
            panic!("Expected IndexExpr view");
        }
    }

    #[test]
    fn test_flatten_view_chain_simple_view() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![3, 4]);

        // 単純なpermute
        let permuted = a.view(a.view.clone().permute(vec![1, 0]));

        let (view, srcs) = permuted.flatten_view_chain();

        // 形状は [4, 3]
        assert_eq!(view.shape(), &[Expr::from(4), Expr::from(3)]);
        // srcは元の入力
        assert_eq!(srcs.len(), 1);
    }

    #[test]
    fn test_flatten_view_chain_gather_then_permute() {
        let mut graph = Graph::new();
        let input = graph.input("input", DType::F32, vec![3, 4, 5]);
        let index = graph.input("index", DType::I32, vec![3, 2, 5]);

        // gather → permute
        let gathered = input.gather(1, &index);
        let permuted = gathered.view(gathered.view.clone().permute(vec![1, 0, 2]));

        let (view, srcs) = permuted.flatten_view_chain();

        // 形状は [2, 3, 5] (permuted)
        assert_eq!(view.shape(), &[Expr::from(2), Expr::from(3), Expr::from(5)]);
        // srcは [input, index]
        assert_eq!(srcs.len(), 2);
        // LoadIndexが含まれている
        assert!(view.contains_load_index());
    }

    #[test]
    fn test_shift_load_index() {
        let expr = Expr::LoadIndex {
            src_index: 1,
            offset_expr: Box::new(Expr::Idx(0) + Expr::Idx(1)),
        };

        let shifted = expr.shift_load_index(2);

        if let Expr::LoadIndex { src_index, .. } = shifted {
            assert_eq!(src_index, 3);
        } else {
            panic!("Expected LoadIndex");
        }
    }

    #[test]
    fn test_view_compose_linear() {
        // contiguous → permute のcompose
        let inner = View::contiguous(vec![3, 4]);
        let outer = inner.clone().permute(vec![1, 0]);

        let composed = View::compose(&outer, &inner);

        // 形状は outer の形状
        assert_eq!(composed.shape(), &[Expr::from(4), Expr::from(3)]);
    }

    #[test]
    fn test_view_shift_load_index() {
        let mut graph = Graph::new();
        let input = graph.input("input", DType::F32, vec![3, 4]);
        let index = graph.input("index", DType::I32, vec![3, 2]);

        let gathered = input.gather(1, &index);

        // Viewをシフト
        let shifted_view = gathered.view.shift_load_index(1);

        if let View::IndexExpr { index_expr, .. } = shifted_view {
            // LoadIndexのsrc_indexが1増えているはず
            fn check_shifted(expr: &Expr) -> bool {
                match expr {
                    Expr::LoadIndex { src_index, .. } => *src_index == 2, // 1 + 1 = 2
                    Expr::Add(l, r) | Expr::Mul(l, r) => check_shifted(l) || check_shifted(r),
                    _ => false,
                }
            }
            assert!(check_shifted(&index_expr));
        } else {
            panic!("Expected IndexExpr");
        }
    }
}
