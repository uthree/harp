use crate::ast::Literal;
use crate::graph::shape::View;
use crate::graph::{DType, GraphNode, GraphNodeData};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::rc::Rc;

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

// DTypeの推論：両方が同じならそれを使う、片方がUnknownなら他方を使う
pub fn infer_dtype(dtype1: &DType, dtype2: &DType) -> DType {
    match (dtype1, dtype2) {
        (DType::Unknown, d) | (d, DType::Unknown) => d.clone(),
        (d1, d2) if std::mem::discriminant(d1) == std::mem::discriminant(d2) => d1.clone(),
        _ => DType::Unknown, // 異なる型の場合はUnknown
    }
}

// Viewの推論：完全に同じshapeのみを許可
// shapeの変更は明示的に行う必要がある（expand, unsqueezeなどを使用）
pub fn infer_view(view1: &View, view2: &View) -> View {
    let shape1 = view1.shape();
    let shape2 = view2.shape();

    // 両方が同じshapeの場合のみ許可
    if shape1 == shape2 {
        return View::contiguous(shape1.to_vec());
    }

    // 異なるshapeの場合はエラー
    panic!(
        "Shape mismatch: {:?} and {:?}. Shape transformations must be explicit (use expand, unsqueeze, etc.)",
        shape1, shape2
    );
}

// 演算子のトレイトを実装

// Add: a + b
impl Add for GraphNode {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let dtype = infer_dtype(&self.dtype, &rhs.dtype);
        let view = infer_view(&self.view, &rhs.view);
        GraphNode(Rc::new(GraphNodeData {
            dtype,
            op: GraphOp::Elementwise(ElementwiseOp::Add),
            src: vec![self, rhs],
            view,
            axis_strategies: None,
        }))
    }
}

// Mul: a * b
impl Mul for GraphNode {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let dtype = infer_dtype(&self.dtype, &rhs.dtype);
        let view = infer_view(&self.view, &rhs.view);
        GraphNode(Rc::new(GraphNodeData {
            dtype,
            op: GraphOp::Elementwise(ElementwiseOp::Mul),
            src: vec![self, rhs],
            view,
            axis_strategies: None,
        }))
    }
}

// Neg: -a
impl Neg for GraphNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode(Rc::new(GraphNodeData {
            dtype,
            op: GraphOp::Elementwise(ElementwiseOp::Neg),
            src: vec![self],
            view,
            axis_strategies: None,
        }))
    }
}

// Sub: a - b = a + (-b)
impl Sub for GraphNode {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

// Div: a / b = a * recip(b)
impl Div for GraphNode {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * recip(rhs)
    }
}

// Rem: a % b
impl Rem for GraphNode {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        let dtype = infer_dtype(&self.dtype, &rhs.dtype);
        let view = infer_view(&self.view, &rhs.view);
        GraphNode(Rc::new(GraphNodeData {
            dtype,
            op: GraphOp::Elementwise(ElementwiseOp::Rem),
            src: vec![self, rhs],
            view,
            axis_strategies: None,
        }))
    }
}

// ヘルパー関数: Recip (逆数)
pub fn recip(node: GraphNode) -> GraphNode {
    let view = node.view.clone();
    let dtype = node.dtype.clone();
    GraphNode(Rc::new(GraphNodeData {
        dtype,
        op: GraphOp::Elementwise(ElementwiseOp::Recip),
        src: vec![node],
        view,
        axis_strategies: None,
    }))
}

// ヘルパー関数: Max
pub fn max(lhs: GraphNode, rhs: GraphNode) -> GraphNode {
    let dtype = infer_dtype(&lhs.dtype, &rhs.dtype);
    let view = infer_view(&lhs.view, &rhs.view);
    GraphNode(Rc::new(GraphNodeData {
        dtype,
        op: GraphOp::Elementwise(ElementwiseOp::Max),
        src: vec![lhs, rhs],
        view,
        axis_strategies: None,
    }))
}
