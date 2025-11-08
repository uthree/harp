use crate::ast::Literal;
use crate::graph::shape::View;
use crate::graph::{CumulativeStrategy, DType, ElementwiseStrategy, GraphNode, ReduceStrategy};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

#[derive(Debug, Clone)]
pub enum GraphOp {
    Input,          // 入力ノード
    Const(Literal), // 定数ノード, shape=[], ndim=0のスカラーを初期化する。
    View(View),     // Viewを変更する
    Contiguous {
        elementwise_strategies: Option<Vec<ElementwiseStrategy>>,
    }, // Viewに従って要素を並べ直す。
    Elementwise {
        op: ElementwiseOp,
        elementwise_strategies: Option<Vec<ElementwiseStrategy>>,
    }, // 要素ごとに演算を行う
    Reduce {
        op: ReduceOp,
        axis: usize,
        reduce_strategy: Option<ReduceStrategy>,
    }, // 縮約
    Cumulative {
        cumulative_strategy: Option<CumulativeStrategy>,
    }, // 累積
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceOp {
    Add, // 合計
    Mul, // 積
    Max, // 最大値
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
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Add,
                elementwise_strategies: None,
            },
            vec![self, rhs],
            view,
        )
    }
}

// Mul: a * b
impl Mul for GraphNode {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let dtype = infer_dtype(&self.dtype, &rhs.dtype);
        let view = infer_view(&self.view, &rhs.view);
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Mul,
                elementwise_strategies: None,
            },
            vec![self, rhs],
            view,
        )
    }
}

// Neg: -a
impl Neg for GraphNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Neg,
                elementwise_strategies: None,
            },
            vec![self],
            view,
        )
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
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        // Division is implemented as multiplication by reciprocal: a / b = a * recip(b)
        self * recip(rhs)
    }
}

// Rem: a % b
impl Rem for GraphNode {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        let dtype = infer_dtype(&self.dtype, &rhs.dtype);
        let view = infer_view(&self.view, &rhs.view);
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Rem,
                elementwise_strategies: None,
            },
            vec![self, rhs],
            view,
        )
    }
}

// ヘルパー関数: Recip (逆数)
pub fn recip(node: GraphNode) -> GraphNode {
    let view = node.view.clone();
    let dtype = node.dtype.clone();
    GraphNode::new(
        dtype,
        GraphOp::Elementwise {
            op: ElementwiseOp::Recip,
            elementwise_strategies: None,
        },
        vec![node],
        view,
    )
}

// ヘルパー関数: Max
pub fn max(lhs: GraphNode, rhs: GraphNode) -> GraphNode {
    let dtype = infer_dtype(&lhs.dtype, &rhs.dtype);
    let view = infer_view(&lhs.view, &rhs.view);
    GraphNode::new(
        dtype,
        GraphOp::Elementwise {
            op: ElementwiseOp::Max,
            elementwise_strategies: None,
        },
        vec![lhs, rhs],
        view,
    )
}

// ヘルパー関数: Reduce（汎用）
pub fn reduce(node: GraphNode, op: ReduceOp, axis: usize) -> GraphNode {
    let dtype = node.dtype.clone();
    let view = node.view.clone();

    // 指定された軸を縮約した新しいViewを作成
    let mut new_shape = view.shape().to_vec();
    if axis >= new_shape.len() {
        panic!(
            "Reduce: axis {} is out of bounds for shape {:?}",
            axis, new_shape
        );
    }
    new_shape.remove(axis);
    let reduced_view = View::contiguous(new_shape);

    GraphNode::new(
        dtype,
        GraphOp::Reduce {
            op,
            axis,
            reduce_strategy: None,
        },
        vec![node],
        reduced_view,
    )
}

// ヘルパー関数: Reduce Sum（指定軸の合計）
pub fn reduce_sum(node: GraphNode, axis: usize) -> GraphNode {
    reduce(node, ReduceOp::Add, axis)
}

// ヘルパー関数: Reduce Mul（指定軸の積）
pub fn reduce_mul(node: GraphNode, axis: usize) -> GraphNode {
    reduce(node, ReduceOp::Mul, axis)
}

// ヘルパー関数: Reduce Max（指定軸の最大値）
pub fn reduce_max(node: GraphNode, axis: usize) -> GraphNode {
    reduce(node, ReduceOp::Max, axis)
}
