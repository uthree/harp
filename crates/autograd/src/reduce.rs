use std::ops;

use crate::traits::{Expand, GradFn, Max, Sum};
use crate::variable::Variable;

// ============================================================================
// SumBackward (総和の逆伝播)
// ============================================================================

/// 総和の勾配関数
/// y = sum(x, axis) の場合、∂L/∂x = expand(∂L/∂y, axis)
pub struct SumBackward<T: 'static> {
    input: Variable<T>,
    axis: usize,
    /// 拡張時に必要なサイズ（縮約前の軸のサイズ）
    size: usize,
}

impl<T: 'static> SumBackward<T> {
    /// 指定した軸で総和を取る勾配関数を作成
    /// size: 縮約前の軸のサイズ（逆伝播での拡張に使用）
    pub fn new(input: Variable<T>, axis: usize, size: usize) -> Self {
        Self { input, axis, size }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// 縮約前の軸のサイズを取得
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<T> GradFn<Variable<T>> for SumBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Expand + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 総和の勾配: 出力の勾配を入力の形状に拡張
        let expanded = grad_y.value().expand(self.axis, self.size);
        self.input.backward_with(Variable::new(expanded));
    }
}

// ============================================================================
// ExpandBackward (拡張の逆伝播)
// ============================================================================

/// 拡張の勾配関数（SumBackward の逆操作）
/// y = expand(x, axis, size) の場合、∂L/∂x = sum(∂L/∂y, axis)
pub struct ExpandBackward<T: 'static> {
    input: Variable<T>,
    axis: usize,
}

impl<T: 'static> ExpandBackward<T> {
    /// 指定した軸で拡張する勾配関数を作成
    pub fn new(input: Variable<T>, axis: usize) -> Self {
        Self { input, axis }
    }

    /// 拡張した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl<T> GradFn<Variable<T>> for ExpandBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Sum + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 拡張の勾配: 出力の勾配を縮約
        let reduced = grad_y.value().sum(self.axis);
        self.input.backward_with(Variable::new(reduced));
    }
}

// ============================================================================
// ProdBackward (総乗の逆伝播)
// ============================================================================

/// 総乗の勾配関数
/// y = prod(x, axis) の場合、∂L/∂x_i = ∂L/∂y * y / x_i
///
/// 逆伝播時に入力値と出力値が必要なため、順伝播時にキャッシュします。
pub struct ProdBackward<T: 'static> {
    input: Variable<T>,
    input_value: T,
    output_value: T,
    axis: usize,
    size: usize,
}

impl<T: Clone + 'static> ProdBackward<T> {
    /// 指定した軸で総乗を取る勾配関数を作成
    pub fn new(input: Variable<T>, output_value: T, axis: usize, size: usize) -> Self {
        let input_value = input.value();
        Self {
            input,
            input_value,
            output_value,
            axis,
            size,
        }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// 順伝播時の入力値を取得
    pub fn input_value(&self) -> &T {
        &self.input_value
    }

    /// 順伝播時の出力値を取得
    pub fn output_value(&self) -> &T {
        &self.output_value
    }
}

impl<T> GradFn<Variable<T>> for ProdBackward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Expand
        + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 総乗の勾配: ∂L/∂x_i = ∂L/∂y * y / x_i
        // grad_y を拡張し、output / input を掛ける
        let expanded_grad = grad_y.value().expand(self.axis, self.size);
        let expanded_output = self.output_value.clone().expand(self.axis, self.size);
        let grad_val = expanded_grad * expanded_output / self.input_value.clone();
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// MaxBackward (最大値の逆伝播)
// ============================================================================

/// 最大値の勾配関数
/// y = max(x, axis) の場合、∂L/∂x_i = ∂L/∂y if x_i == max else 0
///
/// 逆伝播時に入力値が必要なため（最大値の位置を特定するため）、
/// 順伝播時にキャッシュします。
pub struct MaxBackward<T: 'static> {
    input: Variable<T>,
    input_value: T,
    output_value: T,
    axis: usize,
    size: usize,
}

impl<T: Clone + 'static> MaxBackward<T> {
    /// 指定した軸で最大値を取る勾配関数を作成
    pub fn new(input: Variable<T>, output_value: T, axis: usize, size: usize) -> Self {
        let input_value = input.value();
        Self {
            input,
            input_value,
            output_value,
            axis,
            size,
        }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// 順伝播時の入力値を取得
    pub fn input_value(&self) -> &T {
        &self.input_value
    }

    /// 順伝播時の出力値を取得（最大値）
    pub fn output_value(&self) -> &T {
        &self.output_value
    }
}

impl<T> GradFn<Variable<T>> for MaxBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + Max + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 最大値の勾配: 最大値の位置にのみ勾配を伝播
        let grad_val = T::max_grad(
            &grad_y.value(),
            &self.input_value,
            &self.output_value,
            self.axis,
            self.size,
        );
        self.input.backward_with(Variable::new(grad_val));
    }
}
