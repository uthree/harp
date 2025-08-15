use crate::graph::shape::expr::Expr;
use std::collections::HashSet;

// ShapeTrackerの責務は、テンソルの各次元の添字からメモリオフセットへの変換を数式(ExprやAstNode)として表現することです。
// 簡略化のためASTではなくshape::Exprで計算されますが、ExprはASTに簡単に変換することができるので実用上の問題は特にありません。

#[derive(Debug, Clone, PartialEq)]
pub enum View {
    // 線形な処理で表現可能な場合
    Linear {
        shape: Vec<Expr>,   // 論理的なテンソルのサイズ
        strides: Vec<Expr>, // 各次元の添え字の係数
        offset: Expr,       // オフセット
    },
    // 非線形な場合はとりあえず保留する。
}

impl View {
    pub fn new_continuous(shape: Vec<Expr>) -> Self {
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
                    shape: shape,
                    strides: strides,
                    offset: offset,
                }
            }
        }
    }
}
