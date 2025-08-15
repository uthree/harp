use crate::graph::shape::expr::Expr;

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
}

impl View {
    fn new_continuous(shape: Vec<Expr>) -> Self {
        let mut strides = vec![Expr::from(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1].clone() * shape[i + 1].clone();
        }
        View::Linear {
            shape,
            strides,
            offset: Expr::from(0),
        }
    }
}
