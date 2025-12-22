//! 形状変更のプリミティブ演算（Reshape）
//!
//! テンソルの形状を変更する操作を提供します。
//! 要素数は保存される必要があります。

use std::ops;

use super::shape::Shape;
use crate::differentiable::Differentiable;
use crate::traits::GradFn;

// ============================================================================
// Reshape トレイト
// ============================================================================

/// 形状変更を表すトレイト
/// 要素数は保存される必要がある（例: [2, 3] → [3, 2] or [6] or [1, 6] など）
pub trait Reshape: Sized {
    /// 形状を変更
    fn reshape(&self, new_shape: &[usize]) -> Self;
}

// ============================================================================
// ReshapeBackward (形状変更の逆伝播)
// ============================================================================

/// 形状変更の勾配関数
/// y = reshape(x, new_shape) の場合、∂L/∂x = reshape(∂L/∂y, original_shape)
pub struct ReshapeBackward<T: 'static> {
    input: Differentiable<T>,
    /// 元の形状（逆伝播時に戻す）
    original_shape: Vec<usize>,
}

impl<T: 'static> ReshapeBackward<T> {
    pub fn new(input: Differentiable<T>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }

    /// 元の形状を取得
    pub fn original_shape(&self) -> &[usize] {
        &self.original_shape
    }
}

impl<T> GradFn<Differentiable<T>> for ReshapeBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Reshape + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // 形状変更の勾配: 元の形状に戻す
        let requires_grad = grad_y.requires_grad();
        let grad_x = grad_y.value().reshape(&self.original_shape);
        self.input
            .backward_with(Differentiable::new_with_requires_grad(
                grad_x,
                requires_grad,
            ));
    }
}

// ============================================================================
// Variable<T> への実装
// ============================================================================

impl<T> Differentiable<T>
where
    T: Clone + ops::Add<T, Output = T> + Reshape + Shape + 'static,
{
    /// 形状を変更
    pub fn reshape(&self, new_shape: &[usize]) -> Differentiable<T> {
        let original_shape = self.value().shape().to_vec();
        let output = self.value().reshape(new_shape);
        if self.requires_grad() {
            Differentiable::with_grad_fn(
                output,
                Box::new(ReshapeBackward::new(self.clone(), original_shape)),
            )
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}
