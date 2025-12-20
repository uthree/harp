use std::marker::PhantomData;
use std::ops;

use crate::traits::{GradFn, GradInto};
use crate::variable::Variable;

// ============================================================================
// 加算の勾配関数
// ============================================================================

/// 加算の勾配関数
/// z = lhs + rhs の場合、∂L/∂lhs = ∂L/∂z, ∂L/∂rhs = ∂L/∂z
pub struct Add<L: 'static, R: 'static, O: 'static> {
    lhs: Variable<L>,
    rhs: Variable<R>,
    _output: PhantomData<O>,
}

impl<L: 'static, R: 'static, O: 'static> Add<L, R, O> {
    pub fn new(lhs: Variable<L>, rhs: Variable<R>) -> Self {
        Self {
            lhs,
            rhs,
            _output: PhantomData,
        }
    }
}

impl<L, R, O> GradFn<Variable<O>> for Add<L, R, O>
where
    L: Clone + ops::Add<L, Output = L> + 'static,
    R: Clone + ops::Add<R, Output = R> + 'static,
    O: Clone + 'static,
    Variable<O>: GradInto<Variable<L>> + GradInto<Variable<R>> + Clone,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 加算の勾配: 両方に同じ勾配を伝播（型変換付き）
        self.lhs.backward_with(grad_y.clone().gradient_into());
        self.rhs.backward_with(grad_y.gradient_into());
    }
}

// ============================================================================
// 乗算の勾配関数
// ============================================================================

/// 乗算の勾配関数
/// z = lhs * rhs の場合、∂L/∂lhs = ∂L/∂z * rhs, ∂L/∂rhs = ∂L/∂z * lhs
pub struct Mul<L: 'static, R: 'static, O: 'static> {
    lhs: Variable<L>,
    rhs: Variable<R>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    lhs_value: L,
    rhs_value: R,
    _output: PhantomData<O>,
}

impl<L: Clone + 'static, R: Clone + 'static, O: 'static> Mul<L, R, O> {
    pub fn new(lhs: Variable<L>, rhs: Variable<R>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
        let lhs_value = lhs.value();
        let rhs_value = rhs.value();
        Self {
            lhs,
            rhs,
            lhs_value,
            rhs_value,
            _output: PhantomData,
        }
    }
}

impl<L, R, O> GradFn<Variable<O>> for Mul<L, R, O>
where
    L: Clone + ops::Add<L, Output = L> + 'static,
    R: Clone + ops::Add<R, Output = R> + 'static,
    O: Clone + ops::Mul<R, Output = L> + ops::Mul<L, Output = R> + 'static,
    Variable<L>: GradInto<Variable<L>>,
    Variable<R>: GradInto<Variable<R>>,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 乗算の勾配: ∂L/∂lhs = ∂L/∂z * rhs, ∂L/∂rhs = ∂L/∂z * lhs
        let grad_lhs_val = grad_y.value() * self.rhs_value.clone();
        self.lhs.backward_with(Variable::new(grad_lhs_val));

        let grad_rhs_val = grad_y.value() * self.lhs_value.clone();
        self.rhs.backward_with(Variable::new(grad_rhs_val));
    }
}

// ============================================================================
// 符号反転の勾配関数
// ============================================================================

/// 符号反転の勾配関数
/// z = -x の場合、∂L/∂x = -∂L/∂z
pub struct Neg<I: 'static, O: 'static> {
    input: Variable<I>,
    _output: PhantomData<O>,
}

impl<I: 'static, O: 'static> Neg<I, O> {
    pub fn new(input: Variable<I>) -> Self {
        Self {
            input,
            _output: PhantomData,
        }
    }
}

impl<I, O> GradFn<Variable<O>> for Neg<I, O>
where
    I: Clone + ops::Add<I, Output = I> + 'static,
    O: Clone + ops::Neg<Output = I> + 'static,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 符号反転の勾配: ∂L/∂x = -∂L/∂z
        let grad_val = -grad_y.value();
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// 逆数の勾配関数
// ============================================================================

/// 逆数の勾配関数
/// z = 1/x の場合、∂L/∂x = -∂L/∂z / x²
pub struct Recip<I: 'static, O: 'static> {
    input: Variable<I>,
    // 順伝播時の値をキャッシュ（逆伝播で使用）
    input_value: I,
    _output: PhantomData<O>,
}

impl<I: Clone + 'static, O: 'static> Recip<I, O> {
    pub fn new(input: Variable<I>) -> Self {
        // 順伝播時の値をコピー（backward 時に必要）
        let input_value = input.value();
        Self {
            input,
            input_value,
            _output: PhantomData,
        }
    }
}

impl<I, O> GradFn<Variable<O>> for Recip<I, O>
where
    I: Clone + ops::Add<I, Output = I> + ops::Mul<I, Output = I> + ops::Neg<Output = I> + 'static,
    O: Clone + ops::Div<I, Output = I> + 'static,
{
    fn backward(&mut self, grad_y: Variable<O>) {
        // 逆数の勾配: ∂L/∂x = -∂L/∂z / x²
        let x = self.input_value.clone();
        let x_squared = x.clone() * x;
        let grad_val = -(grad_y.value() / x_squared);
        self.input.backward_with(Variable::new(grad_val));
    }
}
