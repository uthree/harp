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
}
