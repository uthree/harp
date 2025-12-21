//! 線形代数のプリミティブ演算（Transpose, Matmul）
//!
//! Array2（2次元配列）に限定した型安全な実装を提供します。

use std::ops;

use crate::traits::GradFn;
use crate::variable::Variable;

// ============================================================================
// Transpose トレイト
// ============================================================================

/// 転置を表すトレイト
pub trait Transpose: Sized {
    fn transpose(&self) -> Self;
}

// ============================================================================
// TransposeBackward (転置の逆伝播)
// ============================================================================

/// 転置の勾配関数
/// Y = X^T の場合、∂L/∂X = (∂L/∂Y)^T
pub struct TransposeBackward<T: 'static> {
    input: Variable<T>,
}

impl<T: 'static> TransposeBackward<T> {
    pub fn new(input: Variable<T>) -> Self {
        Self { input }
    }
}

impl<T> GradFn<Variable<T>> for TransposeBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Transpose + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 転置の勾配: (∂L/∂Y)^T
        let requires_grad = grad_y.requires_grad();
        let grad_x = grad_y.value().transpose();
        self.input
            .backward_with(Variable::new_with_requires_grad(grad_x, requires_grad));
    }
}

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
pub struct MatmulBackward<T: 'static> {
    lhs: Variable<T>,
    rhs: Variable<T>,
    lhs_value: T,
    rhs_value: T,
}

impl<T: Clone + 'static> MatmulBackward<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
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

impl<T> GradFn<Variable<T>> for MatmulBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Transpose + Matmul<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        let requires_grad = grad_y.requires_grad();

        // ∂L/∂A = ∂L/∂C @ B^T
        let rhs_t = self.rhs_value.transpose();
        let grad_lhs = grad_y.value().matmul(&rhs_t);
        self.lhs
            .backward_with(Variable::new_with_requires_grad(grad_lhs, requires_grad));

        // ∂L/∂B = A^T @ ∂L/∂C
        let lhs_t = self.lhs_value.transpose();
        let grad_rhs = lhs_t.matmul(&grad_y.value());
        self.rhs
            .backward_with(Variable::new_with_requires_grad(grad_rhs, requires_grad));
    }
}

// ============================================================================
// Variable<T> への実装
// ============================================================================

impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + Transpose + 'static,
{
    /// 転置を計算
    pub fn transpose(&self) -> Variable<T> {
        let output = self.value().transpose();
        if self.requires_grad() {
            Variable::with_grad_fn(output, Box::new(TransposeBackward::new(self.clone())))
        } else {
            Variable::new_no_grad(output)
        }
    }

    /// 転置を計算（エイリアス）
    pub fn t(&self) -> Variable<T> {
        self.transpose()
    }
}

impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + Transpose + Matmul<T, Output = T> + 'static,
{
    /// 行列積を計算
    pub fn matmul(&self, other: &Variable<T>) -> Variable<T> {
        let output = self.value().matmul(&other.value());
        if self.requires_grad() || other.requires_grad() {
            Variable::with_grad_fn(
                output,
                Box::new(MatmulBackward::new(self.clone(), other.clone())),
            )
        } else {
            Variable::new_no_grad(output)
        }
    }
}
