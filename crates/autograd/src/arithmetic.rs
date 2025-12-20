use std::ops;

use crate::traits::GradFn;
use crate::variable::Variable;

// ============================================================================
// AddBackward (加算の逆伝播)
// ============================================================================

/// 加算の勾配関数
/// z = lhs + rhs の場合、∂L/∂lhs = ∂L/∂z, ∂L/∂rhs = ∂L/∂z
pub struct AddBackward<T: 'static> {
    lhs: Variable<T>,
    rhs: Variable<T>,
}

impl<T: 'static> AddBackward<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T> GradFn<Variable<T>> for AddBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 加算の勾配: 両方に同じ勾配を伝播
        self.lhs.backward_with(grad_y.clone());
        self.rhs.backward_with(grad_y);
    }
}

// ============================================================================
// MulBackward (乗算の逆伝播)
// ============================================================================

/// 乗算の勾配関数
/// z = lhs * rhs の場合、∂L/∂lhs = ∂L/∂z * rhs, ∂L/∂rhs = ∂L/∂z * lhs
pub struct MulBackward<T: 'static> {
    lhs: Variable<T>,
    rhs: Variable<T>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    lhs_value: T,
    rhs_value: T,
}

impl<T: Clone + 'static> MulBackward<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
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

impl<T> GradFn<Variable<T>> for MulBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 乗算の勾配: ∂L/∂lhs = ∂L/∂z * rhs, ∂L/∂rhs = ∂L/∂z * lhs
        let grad_lhs_val = grad_y.value() * self.rhs_value.clone();
        self.lhs.backward_with(Variable::new(grad_lhs_val));

        let grad_rhs_val = grad_y.value() * self.lhs_value.clone();
        self.rhs.backward_with(Variable::new(grad_rhs_val));
    }
}

// ============================================================================
// NegBackward (符号反転の逆伝播)
// ============================================================================

/// 符号反転の勾配関数
/// z = -x の場合、∂L/∂x = -∂L/∂z
pub struct NegBackward<T: 'static> {
    input: Variable<T>,
}

impl<T: 'static> NegBackward<T> {
    pub fn new(input: Variable<T>) -> Self {
        Self { input }
    }
}

impl<T> GradFn<Variable<T>> for NegBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Neg<Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 符号反転の勾配: ∂L/∂x = -∂L/∂z
        let grad_val = -grad_y.value();
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// RecipBackward (逆数の逆伝播)
// ============================================================================

/// 逆数の勾配関数
/// z = 1/x の場合、∂L/∂x = -∂L/∂z / x²
pub struct RecipBackward<T: 'static> {
    input: Variable<T>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    input_value: T,
}

impl<T: Clone + 'static> RecipBackward<T> {
    pub fn new(input: Variable<T>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
        let input_value = input.value();
        Self { input, input_value }
    }
}

impl<T> GradFn<Variable<T>> for RecipBackward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + ops::Neg<Output = T>
        + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 逆数の勾配: ∂L/∂x = -∂L/∂z / x²
        let x = self.input_value.clone();
        let x_squared = x.clone() * x;
        let grad_val = -(grad_y.value() / x_squared);
        self.input.backward_with(Variable::new(grad_val));
    }
}
