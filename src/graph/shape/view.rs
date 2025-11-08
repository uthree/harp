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
}
