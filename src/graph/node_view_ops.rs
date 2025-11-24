// GraphNodeのView関連の操作を提供するモジュール

use crate::graph::shape::{Expr, View};
use crate::graph::{GraphNode, ops::GraphOp};

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
    /// let a = graph.input("a")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 4])
    ///     .build();
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

    /// テンソルの形状を変更（reshape）
    ///
    /// 要素数が同じで、現在のViewが連続している場合のみ使用可能です。
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    /// use harp::graph::shape::Expr;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 4])
    ///     .build();
    ///
    /// // (3, 4) -> (12,) にreshape
    /// let flattened = a.reshape(vec![Expr::from(12)]);
    ///
    /// // (3, 4) -> (2, 6) にreshape
    /// let reshaped = a.reshape(vec![Expr::from(2), Expr::from(6)]);
    /// ```
    pub fn reshape(&self, new_shape: Vec<Expr>) -> Self {
        let new_view = self.view.clone().reshape(new_shape);
        self.view(new_view)
    }

    /// ブロードキャストのためにshapeを拡張
    ///
    /// サイズが1の次元を新しいサイズに拡張します。
    /// サイズが1でない次元は変更できません。
    ///
    /// # パニック
    /// - rankが変わる場合
    /// - サイズが1でない次元を変更しようとした場合
    pub fn expand(&self, new_shape: Vec<Expr>) -> Self {
        let new_view = self.view.clone().expand(new_shape);
        self.view(new_view)
    }

    /// 1D unfold操作（スライディングウィンドウ）
    ///
    /// 畳み込みの前処理として、入力から重複するパッチを抽出します。
    ///
    /// # 引数
    /// - `kernel_size`: ウィンドウサイズ
    /// - `stride`: ストライド（スライディングウィンドウの移動距離）
    /// - `dilation`: 膨張率（カーネル要素間の距離）
    /// - `groups`: グループ数（チャネルを分割する数、2D入力のみ）
    ///
    /// # 入出力
    /// - 1D入力の場合: (L,) -> (k, L')
    /// - 2D入力（groups=1）: (C, L) -> (C, k, L')
    /// - 2D入力（groups=g）: (C, L) -> (g, C/g, k, L')
    ///
    /// where:
    /// - effective_kernel_size = (k - 1) * d + 1
    /// - L' = (L - effective_kernel_size) / s + 1
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
    ///
    /// // 通常のunfold: (10,) -> (3, 8)
    /// let unfolded = x.unfold1d(3, 1, 1, 1);
    ///
    /// // dilationあり: (10,) -> (3, 6)
    /// // effective_kernel_size = (3-1)*2+1 = 5
    /// let unfolded_dilated = x.unfold1d(3, 1, 2, 1);
    ///
    /// // グループ畳み込み
    /// let y = graph.input("y")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![6, 10])
    ///     .build();
    /// // (6, 10) -> (2, 3, 3, 8) with groups=2
    /// let grouped = y.unfold1d(3, 1, 1, 2);
    /// ```
    ///
    /// 現在はpadding=0のみサポート
    pub fn unfold1d(
        &self,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        let new_view = self
            .view
            .clone()
            .unfold1d(kernel_size, stride, dilation, groups);
        self.view(new_view)
    }

    /// 2D unfold操作（スライディングウィンドウ）
    ///
    /// 2次元畳み込みの前処理として、入力から重複するパッチを抽出します。
    ///
    /// # 引数
    /// - `kernel_size`: ウィンドウサイズ (kH, kW)
    /// - `stride`: ストライド (sH, sW)
    /// - `dilation`: 膨張率 (dH, dW)
    /// - `groups`: グループ数（3D入力のみ）
    ///
    /// # 入出力
    /// - 2D入力: (H, W) -> (kH, kW, H', W')
    /// - 3D入力（groups=1）: (C, H, W) -> (C, kH, kW, H', W')
    /// - 3D入力（groups=g）: (C, H, W) -> (g, C/g, kH, kW, H', W')
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
    ///
    /// // 通常の2D unfold: (3, 32, 32) -> (3, 3, 3, 30, 30)
    /// let unfolded = x.unfold2d((3, 3), (1, 1), (1, 1), 1);
    ///
    /// // depthwise: (3, 32, 32) -> (3, 1, 3, 3, 30, 30)
    /// let depthwise = x.unfold2d((3, 3), (1, 1), (1, 1), 3);
    /// ```
    ///
    /// 現在はpadding=0のみサポート
    pub fn unfold2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        let new_view = self
            .view
            .clone()
            .unfold2d(kernel_size, stride, dilation, groups);
        self.view(new_view)
    }

    /// 3D unfold操作（スライディングウィンドウ）
    ///
    /// 3次元畳み込みの前処理として、入力から重複するパッチを抽出します。
    ///
    /// # 引数
    /// - `kernel_size`: ウィンドウサイズ (kD, kH, kW)
    /// - `stride`: ストライド (sD, sH, sW)
    /// - `dilation`: 膨張率 (dD, dH, dW)
    /// - `groups`: グループ数（4D入力のみ）
    ///
    /// # 入出力
    /// - 3D入力: (D, H, W) -> (kD, kH, kW, D', H', W')
    /// - 4D入力（groups=1）: (C, D, H, W) -> (C, kD, kH, kW, D', H', W')
    /// - 4D入力（groups=g）: (C, D, H, W) -> (g, C/g, kD, kH, kW, D', H', W')
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![2, 16, 16, 16])
    ///     .build();
    ///
    /// // 3D unfold: (2, 16, 16, 16) -> (2, 3, 3, 3, 14, 14, 14)
    /// let unfolded = x.unfold3d((3, 3, 3), (1, 1, 1), (1, 1, 1), 1);
    /// ```
    ///
    /// 現在はpadding=0のみサポート
    pub fn unfold3d(
        &self,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Self {
        let new_view = self
            .view
            .clone()
            .unfold3d(kernel_size, stride, dilation, groups);
        self.view(new_view)
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
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 32, 32])
    ///     .build();
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
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![10, 20])
    ///     .build();
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

    /// 1D fold操作（unfoldの逆操作、col2im）
    ///
    /// unfoldされたテンソルを元の形状に戻します。
    /// 重複する部分は加算されます。
    ///
    /// # 引数
    /// - `output_size`: 出力サイズ（unfold前のサイズ）
    /// - `kernel_size`: カーネルサイズ
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: unfold1dの出力形状
    /// - 出力: unfold1dの入力形状
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![2, 3, 8])  // unfoldの出力: (C, k, L')
    ///     .build();
    ///
    /// // fold1d: (2, 3, 8) -> (2, 10)
    /// let folded = x.fold1d(vec![10], 3, 1, 1, 1);
    /// ```
    pub fn fold1d(
        &self,
        output_size: Vec<usize>,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        let new_shape: Vec<Expr> = output_size
            .iter()
            .map(|&s| Expr::from(s as isize))
            .collect();
        let new_view = View::contiguous(new_shape);

        Self::new(
            self.dtype.clone(),
            GraphOp::Fold {
                output_size,
                kernel_size: vec![kernel_size],
                stride: vec![stride],
                dilation: vec![dilation],
                groups,
            },
            vec![self.clone()],
            new_view,
        )
    }

    /// 2D fold操作（unfoldの逆操作、col2im）
    ///
    /// # 引数
    /// - `output_size`: 出力サイズ（unfold前のサイズ）
    /// - `kernel_size`: カーネルサイズ (kH, kW)
    /// - `stride`: ストライド (sH, sW)
    /// - `dilation`: 膨張率 (dH, dW)
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: unfold2dの出力形状
    /// - 出力: unfold2dの入力形状
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 3, 3, 30, 30])  // unfoldの出力: (C, kH, kW, H', W')
    ///     .build();
    ///
    /// // fold2d: (3, 3, 3, 30, 30) -> (3, 32, 32)
    /// let folded = x.fold2d(vec![32, 32], (3, 3), (1, 1), (1, 1), 1);
    /// ```
    pub fn fold2d(
        &self,
        output_size: Vec<usize>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        let new_shape: Vec<Expr> = output_size
            .iter()
            .map(|&s| Expr::from(s as isize))
            .collect();
        let new_view = View::contiguous(new_shape);

        Self::new(
            self.dtype.clone(),
            GraphOp::Fold {
                output_size,
                kernel_size: vec![kernel_size.0, kernel_size.1],
                stride: vec![stride.0, stride.1],
                dilation: vec![dilation.0, dilation.1],
                groups,
            },
            vec![self.clone()],
            new_view,
        )
    }

    /// 3D fold操作（unfoldの逆操作、col2im）
    ///
    /// # 引数
    /// - `output_size`: 出力サイズ（unfold前のサイズ）
    /// - `kernel_size`: カーネルサイズ (kD, kH, kW)
    /// - `stride`: ストライド (sD, sH, sW)
    /// - `dilation`: 膨張率 (dD, dH, dW)
    /// - `groups`: グループ数
    ///
    /// # 入出力
    /// - 入力: unfold3dの出力形状
    /// - 出力: unfold3dの入力形状
    pub fn fold3d(
        &self,
        output_size: Vec<usize>,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Self {
        let new_shape: Vec<Expr> = output_size
            .iter()
            .map(|&s| Expr::from(s as isize))
            .collect();
        let new_view = View::contiguous(new_shape);

        Self::new(
            self.dtype.clone(),
            GraphOp::Fold {
                output_size,
                kernel_size: vec![kernel_size.0, kernel_size.1, kernel_size.2],
                stride: vec![stride.0, stride.1, stride.2],
                dilation: vec![dilation.0, dilation.1, dilation.2],
                groups,
            },
            vec![self.clone()],
            new_view,
        )
    }
}
