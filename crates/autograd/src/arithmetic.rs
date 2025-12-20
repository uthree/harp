use crate::{Backward, Variable};
use std::ops;

// 各種算術演算の実装
// 基本演算: Add, Mul, Neg, Recip
// 派生演算は基本演算の組み合わせで実現:
//   Sub: x - y = x + (-y) → Add + Neg
//   Div: x / y = x * (1/y) → Mul + Recip

/// 加算演算の逆伝播用構造体
/// z = x + y の場合、∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z
pub struct Add<T> {
    lhs: Variable<T>,
    rhs: Variable<T>,
}

impl<T> Add<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T> Backward<T> for Add<T>
where
    T: ops::Add<T, Output = T> + Clone,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 加算の勾配: 両方に同じ勾配を伝播
        // z = x + y => ∂L/∂x = ∂L/∂z, ∂L/∂y = ∂L/∂z
        self.lhs.backward_with(grad_y.clone());
        self.rhs.backward_with(grad_y);
    }
}

/// 乗算演算の逆伝播用構造体
/// z = x * y の場合、∂L/∂x = ∂L/∂z * y, ∂L/∂y = ∂L/∂z * x
pub struct Mul<T> {
    lhs: Variable<T>,
    rhs: Variable<T>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    lhs_value: Variable<T>,
    rhs_value: Variable<T>,
}

impl<T: Clone> Mul<T> {
    pub fn new(lhs: Variable<T>, rhs: Variable<T>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
        let lhs_value = Variable::new(lhs.value());
        let rhs_value = Variable::new(rhs.value());
        Self {
            lhs,
            rhs,
            lhs_value,
            rhs_value,
        }
    }
}

impl<T> Backward<T> for Mul<T>
where
    T: ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Clone,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 乗算の勾配: ∂L/∂x = ∂L/∂z * y, ∂L/∂y = ∂L/∂z * x
        let grad_x = &grad_y * &self.rhs_value;
        self.lhs.backward_with(grad_x);

        let grad_y_for_rhs = &grad_y * &self.lhs_value;
        self.rhs.backward_with(grad_y_for_rhs);
    }
}

/// 符号反転演算の逆伝播用構造体
/// z = -x の場合、∂L/∂x = -∂L/∂z
pub struct Neg<T> {
    input: Variable<T>,
}

impl<T> Neg<T> {
    pub fn new(input: Variable<T>) -> Self {
        Self { input }
    }
}

impl<T> Backward<T> for Neg<T>
where
    T: ops::Add<T, Output = T> + ops::Neg<Output = T> + Clone,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 符号反転の勾配: ∂L/∂x = -∂L/∂z
        self.input.backward_with(-grad_y);
    }
}

/// 逆数演算の逆伝播用構造体
/// z = 1/x の場合、∂L/∂x = -∂L/∂z / x²
pub struct Recip<T> {
    input: Variable<T>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    input_value: Variable<T>,
}

impl<T: Clone> Recip<T> {
    pub fn new(input: Variable<T>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
        let input_value = Variable::new(input.value());
        Self { input, input_value }
    }
}

impl<T> Backward<T> for Recip<T>
where
    T: ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + ops::Neg<Output = T>
        + Clone,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 逆数の勾配: ∂L/∂x = -∂L/∂z / x²
        let x_squared = &self.input_value * &self.input_value;
        let grad_x = -(&grad_y / &x_squared);
        self.input.backward_with(grad_x);
    }
}
