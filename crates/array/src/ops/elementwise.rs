//! 要素演算（四則演算、比較演算）
//!
//! `Array`に対する四則演算子（+, -, *, /）と比較演算を提供します。
//! 演算は遅延評価され、計算グラフとして構築されます。

use crate::array::{Array, ArrayElement};
use crate::backend::Backend;
use crate::dim::Dimension;
use harp_core::graph::GraphNode;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// ヘルパー関数
// ============================================================================

/// 二項演算の結果形状を計算（ブロードキャスト）
fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    // スカラーの場合
    if shape1.is_empty() {
        return shape2.to_vec();
    }
    if shape2.is_empty() {
        return shape1.to_vec();
    }

    // 同じ形状の場合
    if shape1 == shape2 {
        return shape1.to_vec();
    }

    // より複雑なブロードキャストは将来実装
    // 現時点では形状が一致するかスカラーの場合のみサポート
    panic!(
        "Shape mismatch for broadcast: {:?} and {:?}",
        shape1, shape2
    );
}

/// 二項演算を適用
fn apply_binary_op<T, D, B, F>(lhs: &Array<T, D, B>, rhs: &Array<T, D, B>, op: F) -> Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
    F: FnOnce(GraphNode, GraphNode) -> GraphNode,
{
    let lhs_node = lhs.graph_node();
    let rhs_node = rhs.graph_node();
    let result_node = op(lhs_node, rhs_node);
    let result_shape = broadcast_shapes(lhs.shape(), rhs.shape());

    Array::from_graph_node(result_node, result_shape)
}

/// スカラーとの二項演算を適用
fn apply_scalar_op<T, D, B, F>(
    arr: &Array<T, D, B>,
    scalar_node: GraphNode,
    op: F,
) -> Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
    F: FnOnce(GraphNode, GraphNode) -> GraphNode,
{
    let arr_node = arr.graph_node();
    let result_node = op(arr_node, scalar_node);

    Array::from_graph_node(result_node, arr.shape().to_vec())
}

// ============================================================================
// Array + Array 演算
// ============================================================================

// Add: Array + Array
impl<T, D, B> Add for Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        apply_binary_op(&self, &rhs, |l, r| l + r)
    }
}

// Add: &Array + &Array
impl<T, D, B> Add for &Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Array<T, D, B>;

    fn add(self, rhs: Self) -> Self::Output {
        apply_binary_op(self, rhs, |l, r| l + r)
    }
}

// Sub: Array - Array
impl<T, D, B> Sub for Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        apply_binary_op(&self, &rhs, |l, r| l - r)
    }
}

// Sub: &Array - &Array
impl<T, D, B> Sub for &Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Array<T, D, B>;

    fn sub(self, rhs: Self) -> Self::Output {
        apply_binary_op(self, rhs, |l, r| l - r)
    }
}

// Mul: Array * Array
impl<T, D, B> Mul for Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        apply_binary_op(&self, &rhs, |l, r| l * r)
    }
}

// Mul: &Array * &Array
impl<T, D, B> Mul for &Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Array<T, D, B>;

    fn mul(self, rhs: Self) -> Self::Output {
        apply_binary_op(self, rhs, |l, r| l * r)
    }
}

// Div: Array / Array
impl<T, D, B> Div for Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        apply_binary_op(&self, &rhs, |l, r| l / r)
    }
}

// Div: &Array / &Array
impl<T, D, B> Div for &Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Array<T, D, B>;

    fn div(self, rhs: Self) -> Self::Output {
        apply_binary_op(self, rhs, |l, r| l / r)
    }
}

// Neg: -Array
impl<T, D, B> Neg for Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let node = self.graph_node();
        let result_node = -node;
        Array::from_graph_node(result_node, self.shape().to_vec())
    }
}

// Neg: -&Array
impl<T, D, B> Neg for &Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    type Output = Array<T, D, B>;

    fn neg(self) -> Self::Output {
        let node = self.graph_node();
        let result_node = -node;
        Array::from_graph_node(result_node, self.shape().to_vec())
    }
}

// ============================================================================
// Array + スカラー 演算 (f32)
// ============================================================================

/// スカラー演算のマクロ
macro_rules! impl_scalar_ops {
    ($scalar:ty) => {
        // Array + scalar
        impl<D, B> Add<$scalar> for Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Self;

            fn add(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(&self, GraphNode::constant(rhs), |l, r| l + r)
            }
        }

        // &Array + scalar
        impl<D, B> Add<$scalar> for &Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn add(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(self, GraphNode::constant(rhs), |l, r| l + r)
            }
        }

        // scalar + Array
        impl<D, B> Add<Array<f32, D, B>> for $scalar
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn add(self, rhs: Array<f32, D, B>) -> Self::Output {
                apply_scalar_op(&rhs, GraphNode::constant(self), |r, l| l + r)
            }
        }

        // Array - scalar
        impl<D, B> Sub<$scalar> for Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Self;

            fn sub(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(&self, GraphNode::constant(rhs), |l, r| l - r)
            }
        }

        // &Array - scalar
        impl<D, B> Sub<$scalar> for &Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn sub(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(self, GraphNode::constant(rhs), |l, r| l - r)
            }
        }

        // scalar - Array
        impl<D, B> Sub<Array<f32, D, B>> for $scalar
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn sub(self, rhs: Array<f32, D, B>) -> Self::Output {
                apply_scalar_op(&rhs, GraphNode::constant(self), |r, l| l - r)
            }
        }

        // Array * scalar
        impl<D, B> Mul<$scalar> for Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Self;

            fn mul(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(&self, GraphNode::constant(rhs), |l, r| l * r)
            }
        }

        // &Array * scalar
        impl<D, B> Mul<$scalar> for &Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn mul(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(self, GraphNode::constant(rhs), |l, r| l * r)
            }
        }

        // scalar * Array
        impl<D, B> Mul<Array<f32, D, B>> for $scalar
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn mul(self, rhs: Array<f32, D, B>) -> Self::Output {
                apply_scalar_op(&rhs, GraphNode::constant(self), |r, l| l * r)
            }
        }

        // Array / scalar
        impl<D, B> Div<$scalar> for Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Self;

            fn div(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(&self, GraphNode::constant(rhs), |l, r| l / r)
            }
        }

        // &Array / scalar
        impl<D, B> Div<$scalar> for &Array<f32, D, B>
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn div(self, rhs: $scalar) -> Self::Output {
                apply_scalar_op(self, GraphNode::constant(rhs), |l, r| l / r)
            }
        }

        // scalar / Array
        impl<D, B> Div<Array<f32, D, B>> for $scalar
        where
            D: Dimension,
            B: Backend,
        {
            type Output = Array<f32, D, B>;

            fn div(self, rhs: Array<f32, D, B>) -> Self::Output {
                apply_scalar_op(&rhs, GraphNode::constant(self), |r, l| l / r)
            }
        }
    };
}

// f32スカラー演算を実装
impl_scalar_ops!(f32);

// ============================================================================
// 参照演算子の追加実装
// ============================================================================

// scalar + &Array
impl<D, B> Add<&Array<f32, D, B>> for f32
where
    D: Dimension,
    B: Backend,
{
    type Output = Array<f32, D, B>;

    fn add(self, rhs: &Array<f32, D, B>) -> Self::Output {
        apply_scalar_op(rhs, GraphNode::constant(self), |r, l| l + r)
    }
}

// scalar - &Array
impl<D, B> Sub<&Array<f32, D, B>> for f32
where
    D: Dimension,
    B: Backend,
{
    type Output = Array<f32, D, B>;

    fn sub(self, rhs: &Array<f32, D, B>) -> Self::Output {
        apply_scalar_op(rhs, GraphNode::constant(self), |r, l| l - r)
    }
}

// scalar * &Array
impl<D, B> Mul<&Array<f32, D, B>> for f32
where
    D: Dimension,
    B: Backend,
{
    type Output = Array<f32, D, B>;

    fn mul(self, rhs: &Array<f32, D, B>) -> Self::Output {
        apply_scalar_op(rhs, GraphNode::constant(self), |r, l| l * r)
    }
}

// scalar / &Array
impl<D, B> Div<&Array<f32, D, B>> for f32
where
    D: Dimension,
    B: Backend,
{
    type Output = Array<f32, D, B>;

    fn div(self, rhs: &Array<f32, D, B>) -> Self::Output {
        apply_scalar_op(rhs, GraphNode::constant(self), |r, l| l / r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes_same() {
        let result = broadcast_shapes(&[3, 4], &[3, 4]);
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        let result = broadcast_shapes(&[], &[3, 4]);
        assert_eq!(result, vec![3, 4]);

        let result2 = broadcast_shapes(&[3, 4], &[]);
        assert_eq!(result2, vec![3, 4]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_broadcast_shapes_mismatch() {
        broadcast_shapes(&[3, 4], &[3, 5]);
    }
}
