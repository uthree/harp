use super::{Expr, View};

impl View {
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
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert!(
                    shape.len() == 1 || shape.len() == 2,
                    "unfold requires 1D (L,) or 2D (C, L) input"
                );
                assert!(dilation >= 1, "dilation must be >= 1");
                assert!(groups >= 1, "groups must be >= 1");

                let ndim = shape.len();
                let l_dim = ndim - 1; // 最後の次元がL

                // 実効カーネルサイズ: effective_kernel_size = (k - 1) * d + 1
                let effective_kernel_size = (kernel_size - 1) * dilation + 1;

                // 出力サイズ: L' = (L - effective_kernel_size) / s + 1
                let l_out = (shape[l_dim].clone() - Expr::from(effective_kernel_size as isize))
                    / Expr::from(stride as isize)
                    + Expr::from(1);

                let (new_shape, new_strides) = if ndim == 1 {
                    // 1D入力: (L,) -> (k, L')
                    assert!(groups == 1, "groups must be 1 for 1D input");

                    let shape = vec![Expr::from(kernel_size as isize), l_out.simplify()];
                    let strides = vec![
                        (Expr::from(dilation as isize) * strides[0].clone()).simplify(), // k: dilation倍
                        (Expr::from(stride as isize) * strides[0].clone()).simplify(),   // L'
                    ];
                    (shape, strides)
                } else {
                    // 2D入力
                    if groups == 1 {
                        // groups=1: (C, L) -> (C, k, L')
                        let shape = vec![
                            shape[0].clone(),                 // C
                            Expr::from(kernel_size as isize), // k
                            l_out.simplify(),                 // L'
                        ];
                        let strides = vec![
                            strides[0].clone(), // C: 元のチャネルstride
                            (Expr::from(dilation as isize) * strides[1].clone()).simplify(), // k: dilation倍
                            (Expr::from(stride as isize) * strides[1].clone()).simplify(),   // L'
                        ];
                        (shape, strides)
                    } else {
                        // groups=g: (C, L) -> (g, C/g, k, L')
                        // チャネル数がgroupsで割り切れることを確認
                        let c_expr = &shape[0];

                        // 定数の場合のみ検証
                        if let Expr::Const(c) = c_expr {
                            assert!(
                                *c % groups as isize == 0,
                                "Number of channels must be divisible by groups"
                            );
                        }

                        let c_per_group =
                            (shape[0].clone() / Expr::from(groups as isize)).simplify();

                        let shape = vec![
                            Expr::from(groups as isize),      // g
                            c_per_group.clone(),              // C/g
                            Expr::from(kernel_size as isize), // k
                            l_out.simplify(),                 // L'
                        ];

                        // strides計算:
                        // グループ: (C/g) * 元のチャネルstride
                        // チャネル: 元のチャネルstride
                        // カーネル: dilation * 元のL stride
                        // 出力位置: stride * 元のL stride
                        let strides = vec![
                            (c_per_group * strides[0].clone()).simplify(), // g
                            strides[0].clone(),                            // C/g
                            (Expr::from(dilation as isize) * strides[1].clone()).simplify(), // k
                            (Expr::from(stride as isize) * strides[1].clone()).simplify(), // L'
                        ];
                        (shape, strides)
                    }
                };

                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
        }
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
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert!(
                    shape.len() == 2 || shape.len() == 3,
                    "unfold2d requires 2D (H, W) or 3D (C, H, W) input"
                );
                assert!(dilation.0 >= 1 && dilation.1 >= 1, "dilation must be >= 1");
                assert!(groups >= 1, "groups must be >= 1");

                let ndim = shape.len();
                let (kh, kw) = kernel_size;
                let (sh, sw) = stride;
                let (dh, dw) = dilation;

                // 実効カーネルサイズ
                let effective_kernel_h = (kh - 1) * dh + 1;
                let effective_kernel_w = (kw - 1) * dw + 1;

                // 出力サイズ
                let h_idx = ndim - 2;
                let w_idx = ndim - 1;
                let h_out = (shape[h_idx].clone() - Expr::from(effective_kernel_h as isize))
                    / Expr::from(sh as isize)
                    + Expr::from(1);
                let w_out = (shape[w_idx].clone() - Expr::from(effective_kernel_w as isize))
                    / Expr::from(sw as isize)
                    + Expr::from(1);

                let (new_shape, new_strides) = if ndim == 2 {
                    // 2D入力: (H, W) -> (kH, kW, H', W')
                    assert!(groups == 1, "groups must be 1 for 2D input");

                    let shape = vec![
                        Expr::from(kh as isize),
                        Expr::from(kw as isize),
                        h_out.simplify(),
                        w_out.simplify(),
                    ];

                    let strides = vec![
                        (Expr::from(dh as isize) * strides[0].clone()).simplify(), // kH
                        (Expr::from(dw as isize) * strides[1].clone()).simplify(), // kW
                        (Expr::from(sh as isize) * strides[0].clone()).simplify(), // H'
                        (Expr::from(sw as isize) * strides[1].clone()).simplify(), // W'
                    ];

                    (shape, strides)
                } else {
                    // 3D入力
                    if groups == 1 {
                        // groups=1: (C, H, W) -> (C, kH, kW, H', W')
                        let shape = vec![
                            shape[0].clone(),
                            Expr::from(kh as isize),
                            Expr::from(kw as isize),
                            h_out.simplify(),
                            w_out.simplify(),
                        ];

                        let strides = vec![
                            strides[0].clone(),                                        // C
                            (Expr::from(dh as isize) * strides[1].clone()).simplify(), // kH
                            (Expr::from(dw as isize) * strides[2].clone()).simplify(), // kW
                            (Expr::from(sh as isize) * strides[1].clone()).simplify(), // H'
                            (Expr::from(sw as isize) * strides[2].clone()).simplify(), // W'
                        ];

                        (shape, strides)
                    } else {
                        // groups=g: (C, H, W) -> (g, C/g, kH, kW, H', W')
                        let c_expr = &shape[0];

                        // 定数の場合のみ検証
                        if let Expr::Const(c) = c_expr {
                            assert!(
                                *c % groups as isize == 0,
                                "Number of channels must be divisible by groups"
                            );
                        }

                        let c_per_group =
                            (shape[0].clone() / Expr::from(groups as isize)).simplify();

                        let shape = vec![
                            Expr::from(groups as isize),
                            c_per_group.clone(),
                            Expr::from(kh as isize),
                            Expr::from(kw as isize),
                            h_out.simplify(),
                            w_out.simplify(),
                        ];

                        let strides = vec![
                            (c_per_group * strides[0].clone()).simplify(), // g
                            strides[0].clone(),                            // C/g
                            (Expr::from(dh as isize) * strides[1].clone()).simplify(), // kH
                            (Expr::from(dw as isize) * strides[2].clone()).simplify(), // kW
                            (Expr::from(sh as isize) * strides[1].clone()).simplify(), // H'
                            (Expr::from(sw as isize) * strides[2].clone()).simplify(), // W'
                        ];

                        (shape, strides)
                    }
                };

                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
        }
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
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert!(
                    shape.len() == 3 || shape.len() == 4,
                    "unfold3d requires 3D (D, H, W) or 4D (C, D, H, W) input"
                );
                assert!(
                    dilation.0 >= 1 && dilation.1 >= 1 && dilation.2 >= 1,
                    "dilation must be >= 1"
                );
                assert!(groups >= 1, "groups must be >= 1");

                let ndim = shape.len();
                let (kd, kh, kw) = kernel_size;
                let (sd, sh, sw) = stride;
                let (dd, dh, dw) = dilation;

                // 実効カーネルサイズ
                let effective_kernel_d = (kd - 1) * dd + 1;
                let effective_kernel_h = (kh - 1) * dh + 1;
                let effective_kernel_w = (kw - 1) * dw + 1;

                // 出力サイズ
                let d_idx = ndim - 3;
                let h_idx = ndim - 2;
                let w_idx = ndim - 1;
                let d_out = (shape[d_idx].clone() - Expr::from(effective_kernel_d as isize))
                    / Expr::from(sd as isize)
                    + Expr::from(1);
                let h_out = (shape[h_idx].clone() - Expr::from(effective_kernel_h as isize))
                    / Expr::from(sh as isize)
                    + Expr::from(1);
                let w_out = (shape[w_idx].clone() - Expr::from(effective_kernel_w as isize))
                    / Expr::from(sw as isize)
                    + Expr::from(1);

                let (new_shape, new_strides) = if ndim == 3 {
                    // 3D入力: (D, H, W) -> (kD, kH, kW, D', H', W')
                    assert!(groups == 1, "groups must be 1 for 3D input");

                    let shape = vec![
                        Expr::from(kd as isize),
                        Expr::from(kh as isize),
                        Expr::from(kw as isize),
                        d_out.simplify(),
                        h_out.simplify(),
                        w_out.simplify(),
                    ];

                    let strides = vec![
                        (Expr::from(dd as isize) * strides[0].clone()).simplify(), // kD
                        (Expr::from(dh as isize) * strides[1].clone()).simplify(), // kH
                        (Expr::from(dw as isize) * strides[2].clone()).simplify(), // kW
                        (Expr::from(sd as isize) * strides[0].clone()).simplify(), // D'
                        (Expr::from(sh as isize) * strides[1].clone()).simplify(), // H'
                        (Expr::from(sw as isize) * strides[2].clone()).simplify(), // W'
                    ];

                    (shape, strides)
                } else {
                    // 4D入力
                    if groups == 1 {
                        // groups=1: (C, D, H, W) -> (C, kD, kH, kW, D', H', W')
                        let shape = vec![
                            shape[0].clone(),
                            Expr::from(kd as isize),
                            Expr::from(kh as isize),
                            Expr::from(kw as isize),
                            d_out.simplify(),
                            h_out.simplify(),
                            w_out.simplify(),
                        ];

                        let strides = vec![
                            strides[0].clone(),                                        // C
                            (Expr::from(dd as isize) * strides[1].clone()).simplify(), // kD
                            (Expr::from(dh as isize) * strides[2].clone()).simplify(), // kH
                            (Expr::from(dw as isize) * strides[3].clone()).simplify(), // kW
                            (Expr::from(sd as isize) * strides[1].clone()).simplify(), // D'
                            (Expr::from(sh as isize) * strides[2].clone()).simplify(), // H'
                            (Expr::from(sw as isize) * strides[3].clone()).simplify(), // W'
                        ];

                        (shape, strides)
                    } else {
                        // groups=g: (C, D, H, W) -> (g, C/g, kD, kH, kW, D', H', W')
                        let c_expr = &shape[0];

                        // 定数の場合のみ検証
                        if let Expr::Const(c) = c_expr {
                            assert!(
                                *c % groups as isize == 0,
                                "Number of channels must be divisible by groups"
                            );
                        }

                        let c_per_group =
                            (shape[0].clone() / Expr::from(groups as isize)).simplify();

                        let shape = vec![
                            Expr::from(groups as isize),
                            c_per_group.clone(),
                            Expr::from(kd as isize),
                            Expr::from(kh as isize),
                            Expr::from(kw as isize),
                            d_out.simplify(),
                            h_out.simplify(),
                            w_out.simplify(),
                        ];

                        let strides = vec![
                            (c_per_group * strides[0].clone()).simplify(), // g
                            strides[0].clone(),                            // C/g
                            (Expr::from(dd as isize) * strides[1].clone()).simplify(), // kD
                            (Expr::from(dh as isize) * strides[2].clone()).simplify(), // kH
                            (Expr::from(dw as isize) * strides[3].clone()).simplify(), // kW
                            (Expr::from(sd as isize) * strides[1].clone()).simplify(), // D'
                            (Expr::from(sh as isize) * strides[2].clone()).simplify(), // H'
                            (Expr::from(sw as isize) * strides[3].clone()).simplify(), // W'
                        ];

                        (shape, strides)
                    }
                };

                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
        }
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
    #[should_panic(expected = "unfold requires 1D (L,) or 2D (C, L) input")]
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
