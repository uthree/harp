use crate::ast::AstNode;
use crate::graph::shape::Expr;
use std::collections::HashSet;

// ShapeTrackerの責務は、テンソルの各次元の添字からメモリオフセットへの変換を数式(ExprやAstNode)として表現することです。
// 簡略化のためASTではなくshape::Exprで計算されますが、ExprはASTに簡単に変換することができるので実用上の問題は特にありません。

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum View {
    // 線形な処理で表現可能な場合
    Linear {
        shape: Vec<Expr>,   // 論理的なテンソルのサイズ
        strides: Vec<Expr>, // 各次元の添え字の係数
        offset: Expr,       // オフセット
    },
    // 非線形な場合は後で実装する
}

impl View {
    pub fn to_physical_index_ast(&self, logical_indices: &[AstNode]) -> AstNode {
        match self {
            View::Linear {
                strides, offset, ..
            } => {
                assert_eq!(logical_indices.len(), strides.len());
                let mut physical_index: Option<AstNode> = None;

                // Add offset if it's not zero
                if !offset.is_zero() {
                    physical_index = Some(offset.to_ast());
                }

                for (index, stride) in logical_indices.iter().zip(strides.iter()) {
                    if stride.is_zero() {
                        continue;
                    }
                    let term = if stride.is_one() {
                        index.clone()
                    } else {
                        index.clone() * stride.to_ast()
                    };

                    if let Some(p_idx) = physical_index {
                        physical_index = Some(p_idx + term);
                    } else {
                        physical_index = Some(term);
                    }
                }
                physical_index.unwrap_or_else(|| AstNode::from(0isize))
            }
        }
    }

    pub fn shape(&self) -> &[Expr] {
        match self {
            View::Linear { shape, .. } => shape,
        }
    }

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
        match self {
            View::Linear { shape, .. } => shape.len(),
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

    pub fn is_contiguous(&self) -> bool {
        match self {
            View::Linear { shape, .. } => *self == View::new_contiguous(shape.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(vec![2, 3, 4], vec![Expr::from(12), Expr::from(4), Expr::from(1)])]
    #[case(vec![10], vec![Expr::from(1)])]
    #[case(vec![], vec![])]
    fn test_new_contiguous(#[case] shape: Vec<isize>, #[case] expected_strides: Vec<Expr>) {
        let view = View::new_contiguous(shape.clone());
        let View::Linear {
            shape: s,
            strides,
            offset,
        } = view;

        let expected_shape: Vec<Expr> = shape.into_iter().map(|d| d.into()).collect();
        assert_eq!(s, expected_shape);
        assert_eq!(strides, expected_strides);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_permute() {
        let view = View::new_contiguous(vec![2, 3, 4]);
        let permuted_view = view.permute(vec![1, 2, 0]);
        let View::Linear { shape, strides, .. } = permuted_view;
        assert_eq!(shape, vec![Expr::from(3), Expr::from(4), Expr::from(2)]);
        assert_eq!(strides, vec![Expr::from(4), Expr::from(1), Expr::from(12)]);
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let view = View::new_contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(1);
        let View::Linear { shape, strides, .. } = unsqueezed.clone();
        assert_eq!(shape, vec![Expr::from(3), 1.into(), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(4), 0.into(), Expr::from(1)]);

        let squeezed = unsqueezed.squeeze(1);
        let View::Linear { shape, strides, .. } = squeezed;
        assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(4), Expr::from(1)]);
    }

    #[test]
    #[should_panic(expected = "can only squeeze an axis of size 1")]
    fn test_squeeze_invalid_axis() {
        let view = View::new_contiguous(vec![3, 4]);
        view.squeeze(0);
    }

    #[test]
    fn test_is_contiguous() {
        let view = View::new_contiguous(vec![2, 3, 4]);
        assert!(view.is_contiguous());

        let permuted_view = view.permute(vec![1, 2, 0]);
        assert!(!permuted_view.is_contiguous());
    }
}
