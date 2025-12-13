use super::{Expr, View};
use crate::graph::conv::ConvParams;

impl View {
    /// N次元unfold操作
    ///
    /// スライディングウィンドウで入力から重複するパッチを抽出します。
    /// 1D/2D/3D unfoldの共通実装です。
    ///
    /// # 引数
    /// - `params`: 畳み込みパラメータ（kernel_size, stride, dilation, groups）
    ///
    /// # 入出力形状
    /// - N次元入力（空間のみ）: (...) -> (k1, k2, ..., L1', L2', ...)
    /// - (N+1)次元入力（チャネル付き, groups=1）: (C, ...) -> (C, k1, k2, ..., L1', L2', ...)
    /// - (N+1)次元入力（チャネル付き, groups=g）: (C, ...) -> (g, C/g, k1, k2, ..., L1', L2', ...)
    ///
    /// 現在はpadding=0のみサポート
    #[allow(clippy::needless_range_loop)]
    pub fn unfold_nd(self, params: &ConvParams) -> Self {
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let spatial_dims = params.ndim();
                let input_ndim = shape.len();

                // 入力は空間のみ（spatial_dims）か、チャネル付き（spatial_dims + 1）
                assert!(
                    input_ndim == spatial_dims || input_ndim == spatial_dims + 1,
                    "unfold{}d requires {}D or {}D input, got {}D",
                    spatial_dims,
                    spatial_dims,
                    spatial_dims + 1,
                    input_ndim
                );

                // dilation >= 1 の検証
                for &d in &params.dilation {
                    assert!(d >= 1, "dilation must be >= 1");
                }
                assert!(params.groups >= 1, "groups must be >= 1");

                let has_channel = input_ndim == spatial_dims + 1;

                // チャネルなしの場合、groups=1のみ
                if !has_channel {
                    assert!(
                        params.groups == 1,
                        "groups must be 1 for {}D input",
                        spatial_dims
                    );
                }

                // 実効カーネルサイズと出力サイズを計算
                let eff_kernel = params.effective_kernel_size();
                let spatial_start = if has_channel { 1 } else { 0 };

                let output_sizes: Vec<Expr> = (0..spatial_dims)
                    .map(|i| {
                        let input_size = &shape[spatial_start + i];
                        (input_size.clone() - Expr::from(eff_kernel[i] as isize))
                            / Expr::from(params.stride[i] as isize)
                            + Expr::from(1)
                    })
                    .map(|e| e.simplify())
                    .collect();

                let (new_shape, new_strides) = if !has_channel {
                    // 空間のみ: (...) -> (k1, k2, ..., L1', L2', ...)
                    let mut new_shape = Vec::with_capacity(spatial_dims * 2);
                    let mut new_strides = Vec::with_capacity(spatial_dims * 2);

                    // カーネルサイズ次元
                    for i in 0..spatial_dims {
                        new_shape.push(Expr::from(params.kernel_size[i] as isize));
                        new_strides.push(
                            (Expr::from(params.dilation[i] as isize) * strides[i].clone())
                                .simplify(),
                        );
                    }
                    // 出力サイズ次元
                    for i in 0..spatial_dims {
                        new_shape.push(output_sizes[i].clone());
                        new_strides.push(
                            (Expr::from(params.stride[i] as isize) * strides[i].clone()).simplify(),
                        );
                    }

                    (new_shape, new_strides)
                } else if params.groups == 1 {
                    // チャネル付き, groups=1: (C, ...) -> (C, k1, k2, ..., L1', L2', ...)
                    let mut new_shape = Vec::with_capacity(1 + spatial_dims * 2);
                    let mut new_strides = Vec::with_capacity(1 + spatial_dims * 2);

                    // チャネル次元
                    new_shape.push(shape[0].clone());
                    new_strides.push(strides[0].clone());

                    // カーネルサイズ次元
                    for i in 0..spatial_dims {
                        new_shape.push(Expr::from(params.kernel_size[i] as isize));
                        new_strides.push(
                            (Expr::from(params.dilation[i] as isize) * strides[1 + i].clone())
                                .simplify(),
                        );
                    }
                    // 出力サイズ次元
                    for i in 0..spatial_dims {
                        new_shape.push(output_sizes[i].clone());
                        new_strides.push(
                            (Expr::from(params.stride[i] as isize) * strides[1 + i].clone())
                                .simplify(),
                        );
                    }

                    (new_shape, new_strides)
                } else {
                    // チャネル付き, groups=g: (C, ...) -> (g, C/g, k1, k2, ..., L1', L2', ...)
                    let c_expr = &shape[0];

                    // 定数の場合のみ検証
                    if let Expr::Const(c) = c_expr {
                        assert!(
                            *c % params.groups as isize == 0,
                            "Number of channels must be divisible by groups"
                        );
                    }

                    let c_per_group =
                        (shape[0].clone() / Expr::from(params.groups as isize)).simplify();

                    let mut new_shape = Vec::with_capacity(2 + spatial_dims * 2);
                    let mut new_strides = Vec::with_capacity(2 + spatial_dims * 2);

                    // グループ次元
                    new_shape.push(Expr::from(params.groups as isize));
                    new_strides.push((c_per_group.clone() * strides[0].clone()).simplify());

                    // グループ内チャネル次元
                    new_shape.push(c_per_group);
                    new_strides.push(strides[0].clone());

                    // カーネルサイズ次元
                    for i in 0..spatial_dims {
                        new_shape.push(Expr::from(params.kernel_size[i] as isize));
                        new_strides.push(
                            (Expr::from(params.dilation[i] as isize) * strides[1 + i].clone())
                                .simplify(),
                        );
                    }
                    // 出力サイズ次元
                    for i in 0..spatial_dims {
                        new_shape.push(output_sizes[i].clone());
                        new_strides.push(
                            (Expr::from(params.stride[i] as isize) * strides[1 + i].clone())
                                .simplify(),
                        );
                    }

                    (new_shape, new_strides)
                };

                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
        }
    }

    /// 1D unfold操作
    ///
    /// スライディングウィンドウで入力から重複するパッチを抽出します。
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
    /// ```
    /// // 通常のunfold
    /// // 入力: [1, 2, 3, 4, 5, 6] (shape: [6])
    /// // kernel_size=3, stride=1, dilation=1
    /// // 出力shape: [3, 4]
    /// // [[1, 2, 3, 4],
    /// //  [2, 3, 4, 5],
    /// //  [3, 4, 5, 6]]
    ///
    /// // dilation=2の場合
    /// // 入力: [1, 2, 3, 4, 5, 6, 7, 8] (shape: [8])
    /// // kernel_size=3, stride=1, dilation=2
    /// // effective_kernel_size = (3-1)*2+1 = 5
    /// // 出力shape: [3, 4]
    /// // [[1, 3, 5, 2],  // 要素間が2ステップ
    /// //  [3, 5, 7, 4],
    /// //  [5, 7, 9, 6]]
    /// ```
    ///
    /// 現在はpadding=0のみサポート
    pub fn unfold1d(
        self,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        self.unfold_nd(&ConvParams::from_1d(kernel_size, stride, dilation, groups))
    }

    /// 2D unfold操作
    ///
    /// 2次元のスライディングウィンドウで入力から重複するパッチを抽出します。
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
    /// where:
    /// - effective_kernel_size_h = (kH - 1) * dH + 1
    /// - effective_kernel_size_w = (kW - 1) * dW + 1
    /// - H' = (H - effective_kernel_size_h) / sH + 1
    /// - W' = (W - effective_kernel_size_w) / sW + 1
    ///
    /// 現在はpadding=0のみサポート
    pub fn unfold2d(
        self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        self.unfold_nd(&ConvParams::from_2d(kernel_size, stride, dilation, groups))
    }

    /// 3D unfold操作
    ///
    /// 3次元のスライディングウィンドウで入力から重複するパッチを抽出します。
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
    /// where:
    /// - effective_kernel_size_d = (kD - 1) * dD + 1
    /// - effective_kernel_size_h = (kH - 1) * dH + 1
    /// - effective_kernel_size_w = (kW - 1) * dW + 1
    /// - D' = (D - effective_kernel_size_d) / sD + 1
    /// - H' = (H - effective_kernel_size_h) / sH + 1
    /// - W' = (W - effective_kernel_size_w) / sW + 1
    ///
    /// 現在はpadding=0のみサポート
    pub fn unfold3d(
        self,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Self {
        self.unfold_nd(&ConvParams::from_3d(kernel_size, stride, dilation, groups))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unfold1d_1d_basic() {
        // 入力: (6,), kernel=3, stride=1, dilation=1
        // 出力: (3, 4)
        // L' = (6 - 3) / 1 + 1 = 4
        let view = View::contiguous(vec![6]);
        let unfolded = view.unfold1d(3, 1, 1, 1);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(1), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_1d_stride() {
        // 入力: (10,), kernel=3, stride=2, dilation=1
        // 出力: (3, 4)
        // L' = (10 - 3) / 2 + 1 = 4
        let view = View::contiguous(vec![10]);
        let unfolded = view.unfold1d(3, 2, 1, 1);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(1), Expr::from(2)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_1d_dilation() {
        // 入力: (8,), kernel=3, stride=1, dilation=2
        // effective_kernel_size = (3-1)*2+1 = 5
        // L' = (8 - 5) / 1 + 1 = 4
        let view = View::contiguous(vec![8]);
        let unfolded = view.unfold1d(3, 1, 2, 1);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                // カーネル内stride: dilation=2
                assert_eq!(strides, vec![Expr::from(2), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_2d_basic() {
        // 入力: (3, 6), kernel=3, stride=1, dilation=1
        // 出力: (3, 3, 4)
        // L' = (6 - 3) / 1 + 1 = 4
        let view = View::contiguous(vec![3, 6]);
        let unfolded = view.unfold1d(3, 1, 1, 1);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(3), Expr::from(4)]);
                // 元のstrides: [6, 1]
                // unfold後: [6, 1, 1]
                assert_eq!(strides, vec![Expr::from(6), Expr::from(1), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_2d_stride() {
        // 入力: (2, 10), kernel=4, stride=2, dilation=1
        // 出力: (2, 4, 4)
        // L' = (10 - 4) / 2 + 1 = 4
        let view = View::contiguous(vec![2, 10]);
        let unfolded = view.unfold1d(4, 2, 1, 1);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(2), Expr::from(4), Expr::from(4)]);
                // 元のstrides: [10, 1]
                // unfold後: [10, 1, 2]
                assert_eq!(strides, vec![Expr::from(10), Expr::from(1), Expr::from(2)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_2d_dilation() {
        // 入力: (2, 12), kernel=3, stride=1, dilation=3
        // effective_kernel_size = (3-1)*3+1 = 7
        // L' = (12 - 7) / 1 + 1 = 6
        let view = View::contiguous(vec![2, 12]);
        let unfolded = view.unfold1d(3, 1, 3, 1);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(2), Expr::from(3), Expr::from(6)]);
                // 元のstrides: [12, 1]
                // unfold後: [12, 3, 1] (カーネル内stride=dilation=3)
                assert_eq!(strides, vec![Expr::from(12), Expr::from(3), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_2d_groups() {
        // 入力: (6, 10), kernel=3, stride=1, dilation=1, groups=2
        // 出力: (2, 3, 3, 8)
        // グループ数=2, チャネル/グループ=3
        let view = View::contiguous(vec![6, 10]);
        let unfolded = view.unfold1d(3, 1, 1, 2);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(
                    shape,
                    vec![Expr::from(2), Expr::from(3), Expr::from(3), Expr::from(8)]
                );
                // 元のstrides: [10, 1]
                // グループstride: 3 * 10 = 30
                // チャネルstride: 10
                // カーネルstride: 1
                // 出力位置stride: 1
                assert_eq!(
                    strides,
                    vec![Expr::from(30), Expr::from(10), Expr::from(1), Expr::from(1)]
                );
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_2d_depthwise() {
        // Depthwise: groups = channels
        // 入力: (4, 10), kernel=3, stride=1, dilation=1, groups=4
        // 出力: (4, 1, 3, 8)
        let view = View::contiguous(vec![4, 10]);
        let unfolded = view.unfold1d(3, 1, 1, 4);

        match unfolded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(
                    shape,
                    vec![Expr::from(4), Expr::from(1), Expr::from(3), Expr::from(8)]
                );
                // グループstride: 1 * 10 = 10 (各チャネルが独立)
                assert_eq!(
                    strides,
                    vec![Expr::from(10), Expr::from(10), Expr::from(1), Expr::from(1)]
                );
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unfold_example() {
        // ドキュメントの例を検証
        // 入力: [1, 2, 3, 4, 5, 6] (shape: [6])
        // kernel_size=3, stride=1, dilation=1
        // 出力shape: [3, 4]
        let view = View::contiguous(vec![6]);
        let unfolded = view.unfold1d(3, 1, 1, 1);

        assert_eq!(unfolded.shape(), &[Expr::from(3), Expr::from(4)]);

        // strides確認: 各次元の進み方
        match unfolded {
            View::Linear { strides, .. } => {
                // strides[0]=1: カーネル内で1ステップ進む
                // strides[1]=1: 出力位置で1ステップ進む
                assert_eq!(strides, vec![Expr::from(1), Expr::from(1)]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "unfold1d requires 1D or 2D input, got 3D")]
    fn test_unfold_invalid_ndim() {
        let view = View::contiguous(vec![2, 3, 4]);
        let _ = view.unfold1d(3, 1, 1, 1); // Should panic: 3D入力
    }

    #[test]
    #[should_panic(expected = "groups must be 1 for 1D input")]
    fn test_unfold_1d_invalid_groups() {
        let view = View::contiguous(vec![10]);
        let _ = view.unfold1d(3, 1, 1, 2); // Should panic: 1D入力でgroups>1
    }

    #[test]
    #[should_panic(expected = "Number of channels must be divisible by groups")]
    fn test_unfold_2d_invalid_groups() {
        let view = View::contiguous(vec![5, 10]);
        let _ = view.unfold1d(3, 1, 1, 2); // Should panic: 5は2で割り切れない
    }
}
