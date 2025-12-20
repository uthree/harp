use std::marker::PhantomData;
use std::ops;

use crate::traits::{GradFn, GradInto};
use crate::variable::Variable;

// ============================================================================
// Sum (総和) の勾配関数
// ============================================================================

/// 総和の勾配関数
/// y = sum(x, axis) の場合、∂L/∂x = expand(∂L/∂y, axis)
///
/// 注意: この勾配関数は出力の勾配を GradientInto で変換して伝播します。
/// 軸に沿ったブロードキャスト（Expand）は計算グラフで明示的に行う必要があります。
/// 複数の軸を縮約する場合は、複数の Sum ノードを連結してください。
pub struct Sum<I: 'static, O: 'static> {
    input: Variable<I>,
    axis: usize,
    _output: PhantomData<O>,
}

impl<I: 'static, O: 'static> Sum<I, O> {
    /// 指定した軸で総和を取る勾配関数を作成
    pub fn new(input: Variable<I>, axis: usize) -> Self {
        Self {
            input,
            axis,
            _output: PhantomData,
        }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl<I, O> GradFn<Variable<O>> for Sum<I, O>
where
    I: Clone + ops::Add<I, Output = I> + 'static,
    O: Clone + 'static,
    Variable<O>: GradInto<Variable<I>>,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 総和の勾配: 出力の勾配を入力に伝播
        // 軸方向のブロードキャスト（Expand）は計算グラフで明示的に行う
        self.input.backward_with(grad_y.gradient_into());
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
pub struct Expand<I: 'static, O: 'static> {
    input: Variable<I>,
    axis: usize,
    _output: PhantomData<O>,
}

impl<I: 'static, O: 'static> Expand<I, O> {
    /// 指定した軸で拡張する勾配関数を作成
    pub fn new(input: Variable<I>, axis: usize) -> Self {
        Self {
            input,
            axis,
            _output: PhantomData,
        }
    }

    /// 拡張した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl<I, O> GradFn<Variable<O>> for Expand<I, O>
where
    I: Clone + ops::Add<I, Output = I> + 'static,
    O: Clone + 'static,
    Variable<O>: GradInto<Variable<I>>,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 拡張の勾配: 出力の勾配を入力に伝播
        // 軸方向の縮約（Sum）は計算グラフで明示的に行う
        self.input.backward_with(grad_y.gradient_into());
    }
}

// ============================================================================
// Prod (総乗) の勾配関数
// ============================================================================

/// 総乗の勾配関数
/// y = prod(x, axis) の場合、∂L/∂x_i = ∂L/∂y * y / x_i
///
/// 注意: 逆伝播時に入力値と出力値が必要なため、順伝播時にキャッシュします。
/// 複数の軸を縮約する場合は、複数の Prod ノードを連結してください。
pub struct Prod<I: 'static, O: 'static> {
    input: Variable<I>,
    input_value: I,
    output_value: O,
    axis: usize,
    _output: PhantomData<O>,
}

impl<I: Clone + 'static, O: Clone + 'static> Prod<I, O> {
    /// 指定した軸で総乗を取る勾配関数を作成
    pub fn new(input: Variable<I>, output_value: O, axis: usize) -> Self {
        let input_value = input.value();
        Self {
            input,
            input_value,
            output_value,
            axis,
            _output: PhantomData,
        }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// 順伝播時の入力値を取得
    pub fn input_value(&self) -> &I {
        &self.input_value
    }

    /// 順伝播時の出力値を取得
    pub fn output_value(&self) -> &O {
        &self.output_value
    }
}

impl<I, O> GradFn<Variable<O>> for Prod<I, O>
where
    I: Clone + ops::Add<I, Output = I> + 'static,
    O: Clone + 'static,
    Variable<O>: GradInto<Variable<I>>,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 総乗の勾配: ∂L/∂x_i = ∂L/∂y * y / x_i
        // 実際の計算は GradInto の実装で行う（input_value, output_value を参照）
        self.input.backward_with(grad_y.gradient_into());
    }
}

// ============================================================================
// Max (最大値) の勾配関数
// ============================================================================

/// 最大値の勾配関数
/// y = max(x, axis) の場合、∂L/∂x_i = ∂L/∂y if x_i == max else 0
///
/// 注意: 逆伝播時に入力値が必要なため（最大値の位置を特定するため）、
/// 順伝播時にキャッシュします。
/// 複数の軸を縮約する場合は、複数の Max ノードを連結してください。
pub struct Max<I: 'static, O: 'static> {
    input: Variable<I>,
    input_value: I,
    axis: usize,
    _output: PhantomData<O>,
}

impl<I: Clone + 'static, O: 'static> Max<I, O> {
    /// 指定した軸で最大値を取る勾配関数を作成
    pub fn new(input: Variable<I>, axis: usize) -> Self {
        let input_value = input.value();
        Self {
            input,
            input_value,
            axis,
            _output: PhantomData,
        }
    }

    /// 縮約した軸を取得
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// 順伝播時の入力値を取得（最大値の位置特定に使用）
    pub fn input_value(&self) -> &I {
        &self.input_value
    }
}

impl<I, O> GradFn<Variable<O>> for Max<I, O>
where
    I: Clone + ops::Add<I, Output = I> + 'static,
    O: Clone + 'static,
    Variable<O>: GradInto<Variable<I>>,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 最大値の勾配: ∂L/∂x_i = ∂L/∂y if x_i == max else 0
        // 実際の計算は GradInto の実装で行う（input_value を参照してマスクを作成）
        self.input.backward_with(grad_y.gradient_into());
    }
}
