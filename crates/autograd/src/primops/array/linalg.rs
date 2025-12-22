//! 線形代数のプリミティブ演算（Matmul）
//!
//! 行列積の型安全な実装を提供します。

use std::ops;

use super::permute::Permute;
use crate::differentiable::Differentiable;
use crate::traits::GradFn;

// ============================================================================
// Matmul トレイト
// ============================================================================

/// 行列積を表すトレイト
/// C = A @ B
pub trait Matmul<Rhs = Self>: Sized {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

// ============================================================================
// MatmulBackward (行列積の逆伝播)
// ============================================================================

/// 行列積の勾配関数
/// C = A @ B の場合:
/// - ∂L/∂A = ∂L/∂C @ B^T
/// - ∂L/∂B = A^T @ ∂L/∂C
///
/// 転置は permute([1, 0]) で表現
pub struct MatmulBackward<T: 'static> {
    lhs: Differentiable<T>,
    rhs: Differentiable<T>,
    lhs_value: T,
    rhs_value: T,
}

impl<T: Clone + 'static> MatmulBackward<T> {
    pub fn new(lhs: Differentiable<T>, rhs: Differentiable<T>) -> Self {
        let lhs_value = lhs.value();
        let rhs_value = rhs.value();
        Self {
            lhs,
            rhs,
            lhs_value,
            rhs_value,
        }
    }
}

impl<T> GradFn<Differentiable<T>> for MatmulBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Permute + Matmul<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        let requires_grad = grad_y.requires_grad();

        // ∂L/∂A = ∂L/∂C @ B^T (B^T = permute(B, [1, 0]))
        let rhs_t = self.rhs_value.permute(&[1, 0]);
        let grad_lhs = grad_y.value().matmul(&rhs_t);
        self.lhs
            .backward_with(Differentiable::new_with_requires_grad(
                grad_lhs,
                requires_grad,
            ));

        // ∂L/∂B = A^T @ ∂L/∂C (A^T = permute(A, [1, 0]))
        let lhs_t = self.lhs_value.permute(&[1, 0]);
        let grad_rhs = lhs_t.matmul(&grad_y.value());
        self.rhs
            .backward_with(Differentiable::new_with_requires_grad(
                grad_rhs,
                requires_grad,
            ));
    }
}

// ============================================================================
// Variable<T> への実装
// ============================================================================

impl<T> Differentiable<T>
where
    T: Clone + ops::Add<T, Output = T> + Permute + Matmul<T, Output = T> + 'static,
{
    /// 行列積を計算
    pub fn matmul(&self, other: &Differentiable<T>) -> Differentiable<T> {
        let output = self.value().matmul(&other.value());
        if self.requires_grad() || other.requires_grad() {
            Differentiable::with_grad_fn(
                output,
                Box::new(MatmulBackward::new(self.clone(), other.clone())),
            )
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}
