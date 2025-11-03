use crate::{ast::Literal, graph::shape::View};

#[derive(Debug, Clone)]
pub enum GraphOp {
    Input,                      // 入力ノード
    Contiguous,                 // Viewに従って要素を並べ直す。
    Const(Literal),             // 定数ノード, shape=[], ndim=0のスカラーを初期化する。
    View(View),                 // Viewを変更する
    Elementwise(ElementwiseOp), // 要素ごとに演算を行う
    Reduce,                     // 縮約
    Cumulative,                 // 累積
}

#[derive(Debug, Clone)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Max,
    Rem,
    Idiv,
    Neg,
    Recip,
}
