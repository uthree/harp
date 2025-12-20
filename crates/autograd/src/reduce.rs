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
