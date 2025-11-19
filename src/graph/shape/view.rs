use crate::graph::shape::Expr;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum View {
    // 線形な処理で表現可能な場合
    Linear {
        shape: Vec<Expr>,   // 論理的なテンソルのサイズ
        strides: Vec<Expr>, // 各次元の添え字の係数
        offset: Expr,       // オフセット
    },
    // TODO: 非線形な場合の処理を実装する
}

impl View {
    pub fn contiguous<E: Into<Expr> + Clone, I: IntoIterator<Item = E>>(shape: I) -> Self {
        let shape: Vec<Expr> = shape.into_iter().map(|e| e.into()).collect();
        if shape.is_empty() {
            return View::Linear {
                shape,
                strides: vec![],
                offset: Expr::from(0),
            };
        }
        let mut strides = vec![Expr::from(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = (strides[i + 1].clone() * shape[i + 1].clone()).simplify();
        }
        View::Linear {
            shape,
            strides,
            offset: Expr::from(0),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn shape(&self) -> &[Expr] {
        match self {
            View::Linear { shape, .. } => shape,
        }
    }

    pub fn permute(self, axes: Vec<usize>) -> Self {
        assert!(self.ndim() == axes.len());
        let axes_set: HashSet<_> = axes.iter().collect();
        assert!(axes_set.len() == axes.len(), "duplicate axis in permute");
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let mut new_shape = vec![];
                let mut new_strides = vec![];
                for axis in axes.iter() {
                    new_shape.push(shape[*axis].clone().simplify());
                    new_strides.push(strides[*axis].clone().simplify());
                }
                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
        }
    }

    pub fn unsqueeze(self, axis: usize) -> Self {
        assert!(axis <= self.ndim());
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let mut shape = shape.clone();
                let mut strides = strides.clone();
                shape.insert(axis, 1.into());
                strides.insert(axis, 0.into());
                View::Linear {
                    shape,
                    strides,
                    offset,
                }
            }
        }
    }

    pub fn squeeze(self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let mut shape = shape.clone();
                let mut strides = strides.clone();
                assert_eq!(shape[axis], 1.into(), "can only squeeze an axis of size 1");
                shape.remove(axis);
                strides.remove(axis);
                View::Linear {
                    shape,
                    strides,
                    offset,
                }
            }
        }
    }

    pub fn flip(self, axis: usize) -> Self {
        assert!(axis < self.ndim(), "axis out of bounds");
        match self {
            View::Linear {
                shape,
                mut strides,
                offset,
            } => {
                // Flip axis by reversing the stride direction
                // New offset = old_offset + (shape[axis] - 1) * strides[axis]
                // New strides[axis] = -strides[axis]
                let new_offset = (offset
                    + (shape[axis].clone() - Expr::from(1)) * strides[axis].clone())
                .simplify();
                strides[axis] = (-strides[axis].clone()).simplify();
                View::Linear {
                    shape,
                    strides,
                    offset: new_offset,
                }
            }
        }
    }

    pub fn expand(self, new_shape: Vec<Expr>) -> Self {
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape.len(), new_shape.len(), "expand must not change rank");
                let mut new_strides = strides.clone();
                for i in 0..shape.len() {
                    if shape[i] != new_shape[i] {
                        assert!(shape[i].is_one(), "can only expand an axis of size 1");
                        new_strides[i] = 0.into();
                    }
                }
                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
        }
    }

    /// Reshapeは連続したViewに対してのみ適用可能
    ///
    /// 要素数が一致する新しいshapeに変換します。
    /// 非連続なViewに対してはpanicします。
    pub fn reshape(self, new_shape: Vec<Expr>) -> Self {
        assert!(
            self.is_contiguous(),
            "reshape can only be applied to contiguous views"
        );

        match self {
            View::Linear { shape, offset, .. } => {
                // 要素数の一致を確認（シンボリック式の場合は実行時にチェックされる）
                // ここでは定数の場合のみチェック
                let old_numel = shape
                    .iter()
                    .fold(Expr::from(1), |acc, s| acc * s.clone())
                    .simplify();
                let new_numel = new_shape
                    .iter()
                    .fold(Expr::from(1), |acc, s| acc * s.clone())
                    .simplify();

                // 定数の場合のみ検証
                if let (Expr::Const(old_val), Expr::Const(new_val)) = (&old_numel, &new_numel) {
                    assert_eq!(
                        old_val, new_val,
                        "reshape requires the number of elements to match"
                    );
                }

                // 新しいshapeで連続したViewを作成（offsetは保持）
                let mut reshaped = View::contiguous(new_shape);
                let View::Linear {
                    offset: ref mut new_offset,
                    ..
                } = reshaped;
                *new_offset = offset;
                reshaped
            }
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            View::Linear { shape, .. } => *self == View::contiguous(shape.clone()),
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
    fn test_contiguous_1d() {
        let view = View::contiguous(vec![10]);
        match view {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(10)]);
                assert_eq!(strides, vec![Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_contiguous_2d() {
        let view = View::contiguous(vec![3, 4]);
        match view {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(4), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_contiguous_3d() {
        let view = View::contiguous(vec![2, 3, 4]);
        match view {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(2), Expr::from(3), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(12), Expr::from(4), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_contiguous_empty() {
        let view = View::contiguous(Vec::<isize>::new());
        match view {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape.len(), 0);
                assert_eq!(strides.len(), 0);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_ndim() {
        let view = View::contiguous(vec![2, 3, 4]);
        assert_eq!(view.ndim(), 3);

        let view = View::contiguous(vec![10]);
        assert_eq!(view.ndim(), 1);

        let view = View::contiguous(Vec::<isize>::new());
        assert_eq!(view.ndim(), 0);
    }

    #[test]
    fn test_shape() {
        let view = View::contiguous(vec![2, 3, 4]);
        assert_eq!(view.shape(), &[Expr::from(2), Expr::from(3), Expr::from(4)]);
    }

    #[test]
    fn test_permute() {
        let view = View::contiguous(vec![2, 3, 4]);
        let permuted = view.permute(vec![2, 0, 1]); // (2, 3, 4) -> (4, 2, 3)

        match permuted {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(4), Expr::from(2), Expr::from(3)]);
                assert_eq!(strides, vec![Expr::from(1), Expr::from(12), Expr::from(4)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_permute_wrong_axes_count() {
        let view = View::contiguous(vec![2, 3, 4]);
        let _ = view.permute(vec![0, 1]); // Should panic: wrong number of axes
    }

    #[test]
    #[should_panic]
    fn test_permute_duplicate_axes() {
        let view = View::contiguous(vec![2, 3, 4]);
        let _ = view.permute(vec![0, 1, 1]); // Should panic: duplicate axes
    }

    #[test]
    fn test_unsqueeze() {
        let view = View::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(1); // (3, 4) -> (3, 1, 4)

        match unsqueezed {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(1), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(4), Expr::from(0), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_unsqueeze_at_beginning() {
        let view = View::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(0); // (3, 4) -> (1, 3, 4)

        match unsqueezed {
            View::Linear { shape, .. } => {
                assert_eq!(shape, vec![Expr::from(1), Expr::from(3), Expr::from(4)]);
            }
        }
    }

    #[test]
    fn test_unsqueeze_at_end() {
        let view = View::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(2); // (3, 4) -> (3, 4, 1)

        match unsqueezed {
            View::Linear { shape, .. } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4), Expr::from(1)]);
            }
        }
    }

    #[test]
    fn test_squeeze() {
        let view = View::contiguous(vec![3, 1, 4]);
        let squeezed = view.squeeze(1); // (3, 1, 4) -> (3, 4)

        match squeezed {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(4), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    #[should_panic(expected = "can only squeeze an axis of size 1")]
    fn test_squeeze_non_one_axis() {
        let view = View::contiguous(vec![3, 4]);
        let _ = view.squeeze(0); // Should panic: axis 0 has size 3, not 1
    }

    #[test]
    fn test_flip() {
        let view = View::contiguous(vec![3, 4]);
        let flipped = view.flip(0); // Flip first axis

        match flipped {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                // Stride for axis 0 should be negated
                assert_eq!(strides[0], Expr::from(-4));
                assert_eq!(strides[1], Expr::from(1));
                // Offset should be (3-1) * 4 = 8
                assert_eq!(offset, Expr::from(8));
            }
        }
    }

    #[test]
    fn test_expand() {
        let view = View::contiguous(vec![1, 4]);
        let expanded = view.expand(vec![Expr::from(3), Expr::from(4)]); // (1, 4) -> (3, 4)

        match expanded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                // Stride for expanded axis should be 0
                assert_eq!(strides[0], Expr::from(0));
                assert_eq!(strides[1], Expr::from(1));
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    #[should_panic(expected = "can only expand an axis of size 1")]
    fn test_expand_non_one_axis() {
        let view = View::contiguous(vec![3, 4]);
        let _ = view.expand(vec![Expr::from(5), Expr::from(4)]); // Should panic
    }

    #[test]
    #[should_panic(expected = "expand must not change rank")]
    fn test_expand_wrong_rank() {
        let view = View::contiguous(vec![1, 4]);
        let _ = view.expand(vec![Expr::from(3), Expr::from(4), Expr::from(5)]); // Should panic
    }

    #[test]
    fn test_is_contiguous() {
        let view = View::contiguous(vec![3, 4]);
        assert!(view.is_contiguous());

        // Permuted view is not contiguous
        let permuted = View::contiguous(vec![3, 4]).permute(vec![1, 0]);
        assert!(!permuted.is_contiguous());
    }

    #[test]
    fn test_complex_operations() {
        // Create a view and apply multiple operations
        let view = View::contiguous(vec![2, 3, 4]);
        let result = view
            .unsqueeze(0) // (1, 2, 3, 4)
            .permute(vec![0, 2, 1, 3]) // (1, 3, 2, 4)
            .squeeze(0); // (3, 2, 4)

        assert_eq!(result.ndim(), 3);
        assert_eq!(
            result.shape(),
            &[Expr::from(3), Expr::from(2), Expr::from(4)]
        );
    }

    #[test]
    fn test_with_symbolic_shapes() {
        // Test with symbolic expressions
        let batch = Expr::Var("batch".to_string());
        let seq_len = Expr::Var("seq_len".to_string());

        let view = View::contiguous(vec![batch.clone(), seq_len.clone()]);

        match view {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape[0], batch);
                assert_eq!(shape[1], seq_len.clone());
                assert_eq!(strides[0], seq_len);
                assert_eq!(strides[1], Expr::from(1));
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_reshape_basic() {
        // (2, 3, 4) -> (6, 4)
        let view = View::contiguous(vec![2, 3, 4]);
        let reshaped = view.reshape(vec![Expr::from(6), Expr::from(4)]);

        match reshaped {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(6), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(4), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_reshape_flatten() {
        // (2, 3, 4) -> (24,)
        let view = View::contiguous(vec![2, 3, 4]);
        let reshaped = view.reshape(vec![Expr::from(24)]);

        match reshaped {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(24)]);
                assert_eq!(strides, vec![Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    fn test_reshape_expand_dims() {
        // (24,) -> (2, 3, 4)
        let view = View::contiguous(vec![24]);
        let reshaped = view.reshape(vec![Expr::from(2), Expr::from(3), Expr::from(4)]);

        match reshaped {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(2), Expr::from(3), Expr::from(4)]);
                assert_eq!(strides, vec![Expr::from(12), Expr::from(4), Expr::from(1)]);
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    #[should_panic(expected = "reshape can only be applied to contiguous views")]
    fn test_reshape_non_contiguous() {
        // Permuted view is not contiguous
        let view = View::contiguous(vec![2, 3, 4]).permute(vec![2, 1, 0]);
        let _ = view.reshape(vec![Expr::from(6), Expr::from(4)]); // Should panic
    }

    #[test]
    #[should_panic(expected = "reshape requires the number of elements to match")]
    fn test_reshape_wrong_numel() {
        let view = View::contiguous(vec![2, 3, 4]);
        let _ = view.reshape(vec![Expr::from(5), Expr::from(5)]); // 24 != 25, should panic
    }

    #[test]
    fn test_reshape_for_tiling() {
        // Tilingのユースケース: (12, 16) -> (4, 3, 4, 4) -> permute -> (4, 4, 3, 4)
        let view = View::contiguous(vec![12, 16]);
        let tiled = view.reshape(vec![
            Expr::from(4),
            Expr::from(3),
            Expr::from(4),
            Expr::from(4),
        ]);

        match &tiled {
            View::Linear { shape, .. } => {
                assert_eq!(
                    shape,
                    &vec![Expr::from(4), Expr::from(3), Expr::from(4), Expr::from(4)]
                );
            }
        }

        // Permute to group tiles: (4, 4, 3, 4) for better cache locality
        let permuted = tiled.permute(vec![0, 2, 1, 3]);
        assert_eq!(
            permuted.shape(),
            &[Expr::from(4), Expr::from(4), Expr::from(3), Expr::from(4)]
        );
    }

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
