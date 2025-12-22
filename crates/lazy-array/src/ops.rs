//! LazyArray の追加演算メソッド
//!
//! autograd primops に対応するメソッドを提供します。

use crate::backend::{ArrayElement, LazyArray};
use crate::dim::Dimension;
use harp_core::graph::shape::Expr;

// ============================================================================
// 要素単位演算
// ============================================================================

impl<D: Dimension> LazyArray<f32, D> {
    /// 逆数を計算 (1/x)
    pub fn recip(&self) -> Self {
        let node = self.graph_node().recip();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 正弦関数
    pub fn sin(&self) -> Self {
        let node = self.graph_node().sin();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 余弦関数
    pub fn cos(&self) -> Self {
        let node = self.graph_node().cos();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 2を底とする指数関数 (2^x)
    pub fn exp2(&self) -> Self {
        let node = self.graph_node().exp2();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 2を底とする対数関数 (log2(x))
    pub fn log2(&self) -> Self {
        let node = self.graph_node().log2();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 自然指数関数 (e^x)
    pub fn exp(&self) -> Self {
        let node = self.graph_node().exp();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 自然対数関数 (ln(x))
    pub fn log(&self) -> Self {
        let node = self.graph_node().log();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 平方根
    pub fn sqrt(&self) -> Self {
        let node = self.graph_node().sqrt();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 逆平方根 (1/sqrt(x))
    pub fn rsqrt(&self) -> Self {
        let node = self.graph_node().rsqrt();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 二項maximum演算
    pub fn maximum(&self, other: &Self) -> Self {
        let node = self.graph_node().max(other.graph_node());
        Self::from_node(node, self.shape().to_vec(), self.device())
    }

    /// 二乗
    pub fn square(&self) -> Self {
        let node = self.graph_node().square();
        Self::from_node(node, self.shape().to_vec(), self.device())
    }
}

// ============================================================================
// 構造変更演算
// ============================================================================

impl<T: ArrayElement, D: Dimension> LazyArray<T, D> {
    /// 指定軸で総和を計算
    pub fn sum(&self, axis: usize) -> Self {
        let node = self.graph_node().reduce_sum(axis);
        let mut new_shape = self.shape().to_vec();
        if axis < new_shape.len() {
            new_shape.remove(axis);
        }
        Self::from_node(node, new_shape, self.device())
    }

    /// 指定軸で総乗を計算
    pub fn prod(&self, axis: usize) -> Self {
        let node = self.graph_node().reduce_mul(axis);
        let mut new_shape = self.shape().to_vec();
        if axis < new_shape.len() {
            new_shape.remove(axis);
        }
        Self::from_node(node, new_shape, self.device())
    }

    /// 指定軸で最大値を計算
    pub fn max_reduce(&self, axis: usize) -> Self {
        let node = self.graph_node().reduce_max(axis);
        let mut new_shape = self.shape().to_vec();
        if axis < new_shape.len() {
            new_shape.remove(axis);
        }
        Self::from_node(node, new_shape, self.device())
    }

    /// 指定軸方向に拡張（ブロードキャスト）
    pub fn expand(&self, axis: usize, size: usize) -> Self {
        // unsqueezeしてからbroadcast_toする
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, size);

        let target_shape: Vec<Expr> = new_shape.iter().map(|&s| Expr::from(s as isize)).collect();

        // まずunsqueezeでaxisに次元1を挿入
        let unsqueezed_view = self.graph_node().view.clone().unsqueeze(axis);
        let unsqueezed = self.graph_node().view(unsqueezed_view);

        // broadcast_toで拡張
        let node = unsqueezed.broadcast_to(target_shape).contiguous();

        Self::from_node(node, new_shape, self.device())
    }

    /// 軸の順序を変更
    pub fn permute(&self, axes: &[usize]) -> Self {
        let new_view = self.graph_node().view.clone().permute(axes.to_vec());
        let node = self.graph_node().view(new_view).contiguous();
        let new_shape: Vec<usize> = axes.iter().map(|&i| self.shape()[i]).collect();
        Self::from_node(node, new_shape, self.device())
    }

    /// 転置（2次元配列用）
    pub fn transpose(&self) -> Self {
        assert!(
            self.shape().len() >= 2,
            "transpose requires at least 2 dimensions"
        );
        let ndim = self.shape().len();
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(ndim - 2, ndim - 1);
        self.permute(&axes)
    }

    /// 形状を変更
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let target_shape: Vec<Expr> = new_shape.iter().map(|&s| Expr::from(s as isize)).collect();
        let node = self.graph_node().reshape(target_shape);
        Self::from_node(node, new_shape, self.device())
    }

    /// サイズ1の軸を削除
    pub fn squeeze(&self, axis: usize) -> Self {
        assert!(
            axis < self.shape().len() && self.shape()[axis] == 1,
            "squeeze: axis {} must have size 1",
            axis
        );
        let new_view = self.graph_node().view.clone().squeeze(axis);
        let node = self.graph_node().view(new_view);
        let mut new_shape = self.shape().to_vec();
        new_shape.remove(axis);
        Self::from_node(node, new_shape, self.device())
    }

    /// 新しいサイズ1の軸を挿入
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let new_view = self.graph_node().view.clone().unsqueeze(axis);
        let node = self.graph_node().view(new_view);
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);
        Self::from_node(node, new_shape, self.device())
    }
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::Dim1;

    #[test]
    fn test_shape_operations() {
        let arr: LazyArray<f32, Dim1> = <LazyArray<f32, Dim1>>::ones([4]);

        // unsqueeze
        let unsqueezed = arr.unsqueeze(0);
        assert_eq!(unsqueezed.shape(), &[1, 4]);

        // squeeze
        let squeezed = unsqueezed.squeeze(0);
        assert_eq!(squeezed.shape(), &[4]);
    }

    #[test]
    fn test_reshape() {
        let arr: LazyArray<f32, Dim1> = <LazyArray<f32, Dim1>>::ones([6]);
        let reshaped = arr.reshape(vec![2, 3]);
        assert_eq!(reshaped.shape(), &[2, 3]);
    }
}
