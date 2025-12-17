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

    /// 指定した軸を繰り返す（サイズ1の軸のみ対応）
    ///
    /// # Arguments
    /// * `axis` - 繰り返す軸のインデックス
    /// * `times` - 繰り返し回数（結果のサイズ）
    ///
    /// # Panics
    /// * 軸が範囲外の場合
    /// * 指定軸のサイズが1でない場合
    pub fn repeat(self, axis: usize, times: impl Into<Expr>) -> Self {
        let times = times.into();
        match self {
            View::Linear {
                mut shape,
                mut strides,
                offset,
            } => {
                assert!(axis < shape.len(), "axis out of bounds");
                assert!(shape[axis].is_one(), "can only repeat an axis of size 1");
                shape[axis] = times;
                strides[axis] = 0.into();
                View::Linear {
                    shape,
                    strides,
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
    fn test_repeat() {
        let view = View::contiguous(vec![1, 4]);
        let expanded = view.repeat(0, 3); // (1, 4) -> (3, 4), axis 0 repeated 3 times

        match expanded {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
                // Stride for repeated axis should be 0
                assert_eq!(strides[0], Expr::from(0));
                assert_eq!(strides[1], Expr::from(1));
                assert_eq!(offset, Expr::from(0));
            }
        }
    }

    #[test]
    #[should_panic(expected = "can only repeat an axis of size 1")]
    fn test_repeat_non_one_axis() {
        let view = View::contiguous(vec![3, 4]);
        let _ = view.repeat(0, 5); // Should panic: axis 0 is size 3, not 1
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn test_repeat_invalid_axis() {
        let view = View::contiguous(vec![1, 4]);
        let _ = view.repeat(5, 3); // Should panic: axis 5 is out of bounds
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
}
