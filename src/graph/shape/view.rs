use crate::graph::shape::Expr;
use std::collections::HashSet;
use std::fmt;

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
    pub fn new_contiguous<E: Into<Expr> + Clone, I: IntoIterator<Item = E>>(shape: I) -> Self {
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

    pub fn is_contiguous(&self) -> bool {
        match self {
            View::Linear { shape, .. } => *self == View::new_contiguous(shape.clone()),
        }
    }

    /// Create a view with custom strides (similar to PyTorch's as_strided).
    ///
    /// This is a low-level operation that allows direct control over shape and strides.
    /// Use with caution as it can create views that access out-of-bounds memory.
    ///
    /// # Arguments
    /// * `new_shape` - The desired shape for the view
    /// * `new_strides` - The strides for each dimension
    ///
    /// # Safety
    /// The caller must ensure that all accessed indices are within the original buffer bounds.
    pub fn as_strided(self, new_shape: Vec<Expr>, new_strides: Vec<Expr>) -> Self {
        assert_eq!(
            new_shape.len(),
            new_strides.len(),
            "shape and strides must have the same length"
        );

        match self {
            View::Linear { offset, .. } => View::Linear {
                shape: new_shape,
                strides: new_strides,
                offset,
            },
        }
    }

    /// Unfold operation: extract sliding local blocks from a tensor (similar to im2col).
    ///
    /// This operation transforms a view by adding a new dimension for sliding windows.
    /// For example, [B, C, L] with window_size=K, stride=S becomes [B, C, L', K]
    /// where L' = (L - K) / S + 1.
    ///
    /// This is the inverse operation of `fold`, which combines overlapping blocks.
    ///
    /// This is useful for implementing convolution via matrix multiplication:
    /// 1. Apply unfold to input: [B, C, L] → [B, C, L', K]
    /// 2. Reshape kernel: [Co, Ci, K] → [Co, Ci, 1, K]
    /// 3. Multiply: [B, C, L', K] * [Co, Ci, 1, K] → [B, Co, Ci, L', K]
    /// 4. Sum over [Ci, K]: → [B, Co, L']
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to create sliding windows
    /// * `window_size` - The size of each window
    /// * `stride` - The stride between windows
    ///
    /// # Example
    /// ```ignore
    /// // Input: [2, 3, 10] (batch=2, channels=3, length=10)
    /// // window_size=3, stride=1
    /// // Output: [2, 3, 8, 3] (batch=2, channels=3, output_length=8, window_size=3)
    /// let view = View::new_contiguous(vec![2, 3, 10]);
    /// let unfolded = view.unfold(2, 3, 1);
    /// ```
    pub fn unfold<E: Into<Expr> + Clone>(self, dim: usize, window_size: E, stride: E) -> Self {
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                assert!(dim < shape.len(), "dimension out of bounds");

                let window_size_expr: Expr = window_size.into();
                let stride_expr: Expr = stride.into();

                // Calculate output length: (L - K) / S + 1
                let output_length = ((shape[dim].clone() - window_size_expr.clone())
                    / stride_expr.clone()
                    + Expr::from(1))
                .simplify();

                // Build new shape: [..., L', K]
                let mut new_shape = shape.clone();
                new_shape[dim] = output_length;
                new_shape.push(window_size_expr.clone());

                // Build new strides
                let mut new_strides = strides.clone();
                // The original stride at dim is multiplied by the window stride
                new_strides[dim] = (strides[dim].clone() * stride_expr).simplify();
                // The new window dimension has stride equal to the original stride at dim
                new_strides.push(strides[dim].clone());

                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
        }
    }

    pub fn elementwise_result_view(&self, other: &View) -> View {
        // スカラーとのbroadcastを処理
        let self_shape = self.shape();
        let other_shape = other.shape();

        // 片方がスカラー(空の形状)の場合は、もう片方の形状を使用
        if self_shape.is_empty() && !other_shape.is_empty() {
            return other.clone();
        }
        if other_shape.is_empty() && !self_shape.is_empty() {
            return self.clone();
        }

        // 両方スカラーまたは同じ形状
        assert_eq!(
            self_shape, other_shape,
            "shapes must match for element-wise operations"
        );

        // strideが一致する場合は既存のviewを再利用
        match (self, other) {
            (
                View::Linear {
                    shape: lhs_shape,
                    strides: lhs_strides,
                    offset: lhs_offset,
                },
                View::Linear {
                    shape: _rhs_shape,
                    strides: rhs_strides,
                    offset: rhs_offset,
                },
            ) => {
                if lhs_strides == rhs_strides && lhs_offset == rhs_offset {
                    self.clone()
                } else {
                    // stride不一致の場合はcontiguousな新しいviewを作成
                    View::new_contiguous(lhs_shape.clone())
                }
            }
        }
    }
}

impl fmt::Display for View {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                // Format shape
                let shape_str = if shape.is_empty() {
                    "[]".to_string()
                } else {
                    let parts: Vec<String> = shape.iter().map(|e| format!("{}", e)).collect();
                    format!("[{}]", parts.join(", "))
                };

                // Check if this is a contiguous view
                let is_contiguous = *self == View::new_contiguous(shape.clone());

                if is_contiguous && offset.is_zero() {
                    // Simple contiguous case
                    write!(f, "{}", shape_str)
                } else {
                    // Show strides and offset for non-contiguous views
                    let strides_str = if strides.is_empty() {
                        "[]".to_string()
                    } else {
                        let parts: Vec<String> = strides.iter().map(|e| format!("{}", e)).collect();
                        format!("[{}]", parts.join(", "))
                    };

                    if offset.is_zero() {
                        write!(f, "{} (strides: {})", shape_str, strides_str)
                    } else {
                        write!(
                            f,
                            "{} (strides: {}, offset: {})",
                            shape_str, strides_str, offset
                        )
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_contiguous() {
        let view = View::new_contiguous(vec![2, 3, 4]);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(2), Expr::from(3), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(12), Expr::from(4), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_permute() {
        let view = View::new_contiguous(vec![2, 3, 4]).permute(vec![2, 0, 1]);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(4), Expr::from(2), Expr::from(3)]);
        assert_eq!(strides, vec![Expr::from(1), Expr::from(12), Expr::from(4)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_unsqueeze() {
        let view = View::new_contiguous(vec![2, 3]).unsqueeze(1);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(2), Expr::from(1), Expr::from(3)]);
        assert_eq!(strides, vec![Expr::from(3), Expr::from(0), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_squeeze() {
        let view = View::new_contiguous(vec![2, 1, 3]).squeeze(1);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(2), Expr::from(3)]);
        assert_eq!(strides, vec![Expr::from(3), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_expand() {
        let n = Expr::Var("N".to_string());
        let view = View::new_contiguous(vec![2, 1, 3]).expand(vec![2.into(), n.clone(), 3.into()]);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(2), n, Expr::from(3)]);
        assert_eq!(strides, vec![Expr::from(3), Expr::from(0), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_is_contiguous() {
        let view1 = View::new_contiguous(vec![2, 3, 4]);
        assert!(view1.is_contiguous());

        let view2 = view1.permute(vec![0, 2, 1]);
        assert!(!view2.is_contiguous());
    }

    #[test]
    #[should_panic(expected = "duplicate axis in permute")]
    fn test_permute_duplicate_axes() {
        View::new_contiguous(vec![2, 3, 4]).permute(vec![0, 1, 1]);
    }

    #[test]
    #[should_panic]
    fn test_permute_wrong_ndim() {
        View::new_contiguous(vec![2, 3, 4]).permute(vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "can only squeeze an axis of size 1")]
    fn test_squeeze_invalid_axis_size() {
        View::new_contiguous(vec![2, 3]).squeeze(1);
    }

    #[test]
    fn test_flip_axis_0() {
        let view = View::new_contiguous(vec![2, 3]).flip(0);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(2), Expr::from(3)]);
        assert_eq!(strides, vec![Expr::from(-3), Expr::from(1)]);
        assert_eq!(offset, Expr::from(3)); // (2-1) * 3 = 3
    }

    #[test]
    fn test_flip_axis_1() {
        let view = View::new_contiguous(vec![2, 3]).flip(1);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(2), Expr::from(3)]);
        assert_eq!(strides, vec![Expr::from(3), Expr::from(-1)]);
        assert_eq!(offset, Expr::from(2)); // (3-1) * 1 = 2
    }

    #[test]
    fn test_flip_multiple_axes() {
        // Flip axis 0, then axis 1
        let view = View::new_contiguous(vec![2, 3, 4]).flip(0).flip(1);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view;
        assert_eq!(shape, vec![Expr::from(2), Expr::from(3), Expr::from(4)]);
        assert_eq!(
            strides,
            vec![Expr::from(-12), Expr::from(-4), Expr::from(1)]
        );
        // offset = 0 + (2-1)*12 + (3-1)*4 = 12 + 8 = 20
        assert_eq!(offset, Expr::from(20));
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn test_flip_invalid_axis() {
        View::new_contiguous(vec![2, 3]).flip(2);
    }

    #[test]
    fn test_as_strided() {
        let view = View::new_contiguous(vec![2, 3, 4]);
        let custom = view.as_strided(
            vec![Expr::from(2), Expr::from(2), Expr::from(2)],
            vec![Expr::from(6), Expr::from(3), Expr::from(1)],
        );
        let View::Linear {
            shape,
            strides,
            offset,
        } = custom;
        assert_eq!(shape, vec![Expr::from(2), Expr::from(2), Expr::from(2)]);
        assert_eq!(strides, vec![Expr::from(6), Expr::from(3), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_unfold_basic() {
        // Input: [2, 3, 10] (B=2, C=3, L=10)
        // window_size=3, stride=1
        // Output: [2, 3, 8, 3] (B=2, C=3, L'=8, K=3)
        // where L' = (10 - 3) / 1 + 1 = 8
        let view = View::new_contiguous(vec![2, 3, 10]);
        let windowed = view.unfold(2, 3, 1);
        let View::Linear {
            shape,
            strides,
            offset,
        } = windowed;

        // Expected shape: [2, 3, 8, 3]
        assert_eq!(
            shape,
            vec![
                Expr::from(2),
                Expr::from(3),
                Expr::from(8), // (10 - 3) / 1 + 1
                Expr::from(3)
            ]
        );

        // Expected strides:
        // - B axis: 30 (original)
        // - C axis: 10 (original)
        // - L' axis: 1 (stride * original stride at dim 2)
        // - K axis: 1 (original stride at dim 2)
        assert_eq!(
            strides,
            vec![
                Expr::from(30),
                Expr::from(10),
                Expr::from(1), // stride=1 * original_stride=1
                Expr::from(1)  // original_stride at dim 2
            ]
        );

        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_unfold_with_stride() {
        // Input: [1, 1, 10] (B=1, C=1, L=10)
        // window_size=3, stride=2
        // Output: [1, 1, 4, 3] (B=1, C=1, L'=4, K=3)
        // where L' = (10 - 3) / 2 + 1 = 4
        let view = View::new_contiguous(vec![1, 1, 10]);
        let windowed = view.unfold(2, 3, 2);
        let View::Linear {
            shape,
            strides,
            offset,
        } = windowed;

        // Expected shape: [1, 1, 4, 3]
        assert_eq!(
            shape,
            vec![
                Expr::from(1),
                Expr::from(1),
                Expr::from(4), // (10 - 3) / 2 + 1
                Expr::from(3)
            ]
        );

        // Expected strides:
        // - B axis: 10 (original)
        // - C axis: 10 (original)
        // - L' axis: 2 (stride=2 * original_stride=1)
        // - K axis: 1 (original stride at dim 2)
        assert_eq!(
            strides,
            vec![
                Expr::from(10),
                Expr::from(10),
                Expr::from(2), // stride=2 * original_stride=1
                Expr::from(1)  // original_stride at dim 2
            ]
        );

        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    #[should_panic(expected = "dimension out of bounds")]
    fn test_unfold_invalid_dim() {
        View::new_contiguous(vec![2, 3]).unfold(3, 2, 1);
    }
}
