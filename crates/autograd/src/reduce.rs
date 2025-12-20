use std::ops;

use crate::traits::GradFn;
use crate::variable::Variable;

// ============================================================================
// Sum (総和) の勾配関数
// ============================================================================

/// 総和の勾配関数
/// y = sum(x, axis) の場合、∂L/∂x = expand(∂L/∂y, axis)
///
/// 軸に沿ったブロードキャスト（Expand）は計算グラフで明示的に行う必要があります。
/// 複数の軸を縮約する場合は、複数の Sum ノードを連結してください。
pub struct Sum<T: 'static> {
    input: Variable<T>,
    axis: usize,
}

impl<T: 'static> Sum<T> {
    /// 指定した軸で総和を取る勾配関数を作成
    pub fn new(input: Variable<T>, axis: usize) -> Self {
        Self { input, axis }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl<T> GradFn<Variable<T>> for Sum<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        self.input.backward_with(grad_y);
    }
}

// ============================================================================
// Expand (拡張) の勾配関数
// ============================================================================

/// 拡張の勾配関数（Sum の逆操作）
/// y = expand(x, axis, size) の場合、∂L/∂x = sum(∂L/∂y, axis)
///
/// expand は縮約された次元を復元する操作で、sum の逆伝播として使用します。
/// 複数の軸を拡張する場合は、複数の Expand ノードを連結してください。
pub struct Expand<T: 'static> {
    input: Variable<T>,
    axis: usize,
}

impl<T: 'static> Expand<T> {
    /// 指定した軸で拡張する勾配関数を作成
    pub fn new(input: Variable<T>, axis: usize) -> Self {
        Self { input, axis }
    }

    /// 拡張した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl<T> GradFn<Variable<T>> for Expand<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        self.input.backward_with(grad_y);
    }
}

// ============================================================================
// Prod (総乗) の勾配関数
// ============================================================================

/// 総乗の勾配関数
/// y = prod(x, axis) の場合、∂L/∂x_i = ∂L/∂y * y / x_i
///
/// 逆伝播時に入力値と出力値が必要なため、順伝播時にキャッシュします。
/// 複数の軸を縮約する場合は、複数の Prod ノードを連結してください。
pub struct Prod<T: 'static> {
    input: Variable<T>,
    input_value: T,
    output_value: T,
    axis: usize,
}

impl<T: Clone + 'static> Prod<T> {
    /// 指定した軸で総乗を取る勾配関数を作成
    pub fn new(input: Variable<T>, output_value: T, axis: usize) -> Self {
        let input_value = input.value();
        Self {
            input,
            input_value,
            output_value,
            axis,
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

impl<T> GradFn<Variable<T>> for Prod<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        self.input.backward_with(grad_y);
    }
}

// ============================================================================
// Max (最大値) の勾配関数
// ============================================================================

/// 最大値の勾配関数
/// y = max(x, axis) の場合、∂L/∂x_i = ∂L/∂y if x_i == max else 0
///
/// 逆伝播時に入力値が必要なため（最大値の位置を特定するため）、
/// 順伝播時にキャッシュします。
/// 複数の軸を縮約する場合は、複数の Max ノードを連結してください。
pub struct Max<T: 'static> {
    input: Variable<T>,
    input_value: T,
    axis: usize,
}

impl<T: Clone + 'static> Max<T> {
    /// 指定した軸で最大値を取る勾配関数を作成
    pub fn new(input: Variable<T>, axis: usize) -> Self {
        let input_value = input.value();
        Self {
            input,
            input_value,
            axis,
        }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// 順伝播時の入力値を取得（最大値の位置特定に使用）
    pub fn input_value(&self) -> &T {
        &self.input_value
    }
}

impl<T> GradFn<Variable<T>> for Max<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        self.input.backward_with(grad_y);
    }
}
