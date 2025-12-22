//! 軸順序変更のプリミティブ演算（Permute）
//!
//! テンソルの軸の順序を入れ替える操作を提供します。
//! Transpose（転置）はこの操作の特殊ケース（2次元で axes=[1,0]）です。

use std::ops;

use crate::differentiable::Differentiable;
use crate::traits::GradFn;

// ============================================================================
// Permute トレイト
// ============================================================================

/// 軸順序変更を表すトレイト
/// axes: 新しい軸の順序（例: [2, 0, 1] で axis2→axis0, axis0→axis1, axis1→axis2）
pub trait Permute: Sized {
    /// 軸の順序を入れ替える
    fn permute(&self, axes: &[usize]) -> Self;
}

/// 逆順列を計算するユーティリティ関数
/// 例: [2, 0, 1] → [1, 2, 0]
pub fn inverse_permutation(axes: &[usize]) -> Vec<usize> {
    let n = axes.len();
    let mut inverse = vec![0; n];
    for (i, &axis) in axes.iter().enumerate() {
        inverse[axis] = i;
    }
    inverse
}

// ============================================================================
// スカラー型の実装（no-op）
// ============================================================================

impl Permute for f32 {
    fn permute(&self, _axes: &[usize]) -> Self {
        *self
    }
}

impl Permute for f64 {
    fn permute(&self, _axes: &[usize]) -> Self {
        *self
    }
}

// ============================================================================
// PermuteBackward (軸順序変更の逆伝播)
// ============================================================================

/// 軸順序変更の勾配関数
/// y = permute(x, axes) の場合、∂L/∂x = permute(∂L/∂y, inverse(axes))
pub struct PermuteBackward<T: 'static> {
    input: Differentiable<T>,
    /// 逆順列（逆伝播時に使用）
    inverse_axes: Vec<usize>,
}

impl<T: 'static> PermuteBackward<T> {
    pub fn new(input: Differentiable<T>, axes: &[usize]) -> Self {
        let inverse_axes = inverse_permutation(axes);
        Self {
            input,
            inverse_axes,
        }
    }

    /// 逆順列を取得
    pub fn inverse_axes(&self) -> &[usize] {
        &self.inverse_axes
    }
}

impl<T> GradFn<Differentiable<T>> for PermuteBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Permute + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // 軸順序変更の勾配: 逆順列を適用
        let requires_grad = grad_y.requires_grad();
        let grad_x = grad_y.value().permute(&self.inverse_axes);
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
    T: Clone + ops::Add<T, Output = T> + Permute + 'static,
{
    /// 軸の順序を入れ替える
    pub fn permute(&self, axes: &[usize]) -> Differentiable<T> {
        let output = self.value().permute(axes);
        if self.requires_grad() {
            Differentiable::with_grad_fn(output, Box::new(PermuteBackward::new(self.clone(), axes)))
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}
