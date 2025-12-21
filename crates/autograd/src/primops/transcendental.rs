//! 超越関数（Sin, Exp2, Log2, Sqrt など）のプリミティブ演算

use std::ops;

use crate::traits::GradFn;
use crate::variable::Variable;

// ============================================================================
// 超越関数トレイト
// ============================================================================

/// 正弦関数を表すトレイト
pub trait Sin: Sized {
    fn sin(&self) -> Self;
}

/// 余弦関数を表すトレイト（primopsとして定義、計算はhlopsで実装可能）
pub trait Cos: Sized {
    fn cos(&self) -> Self;
}

/// 2を底とする対数関数を表すトレイト
pub trait Log2: Sized {
    fn log2(&self) -> Self;
}

/// 2を底とする指数関数を表すトレイト
pub trait Exp2: Sized {
    fn exp2(&self) -> Self;
}

/// 平方根関数を表すトレイト
pub trait Sqrt: Sized {
    fn sqrt(&self) -> Self;
}

/// ln(2) を取得するトレイト
pub trait Ln2 {
    fn ln2() -> Self;
}

/// 定数 2 を取得するトレイト
pub trait Two {
    fn two() -> Self;
}

/// 位相を1/4周期（π/2ラジアン = 90度）シフトするトレイト
/// cos(x) = sin(x + π/2) を実現するための基本演算
pub trait PhaseShiftQuarter {
    fn phase_shift_quarter(&self) -> Self;
}

// ============================================================================
// SinBackward (正弦の逆伝播)
// ============================================================================

/// 正弦の勾配関数
/// y = sin(x) の場合、∂L/∂x = ∂L/∂y * cos(x)
pub struct SinBackward<T: 'static> {
    input: Variable<T>,
    input_value: T,
}

impl<T: Clone + 'static> SinBackward<T> {
    pub fn new(input: Variable<T>) -> Self {
        let input_value = input.value();
        Self { input, input_value }
    }
}

impl<T> GradFn<Variable<T>> for SinBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Cos + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // sin の勾配: ∂L/∂x = ∂L/∂y * cos(x)
        let cos_x = self.input_value.cos();
        let grad_val = grad_y.value() * cos_x;
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// PhaseShiftQuarterBackward (1/4周期シフトの逆伝播)
// ============================================================================

/// 1/4周期シフトの勾配関数（定数加算なので勾配はそのまま通過）
/// y = x + π/2 の場合、∂L/∂x = ∂L/∂y
pub struct PhaseShiftQuarterBackward<T: 'static> {
    input: Variable<T>,
}

impl<T: 'static> PhaseShiftQuarterBackward<T> {
    pub fn new(input: Variable<T>) -> Self {
        Self { input }
    }
}

impl<T> GradFn<Variable<T>> for PhaseShiftQuarterBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // 定数加算の勾配: そのまま通過
        self.input.backward_with(grad_y);
    }
}

// ============================================================================
// Log2Backward (2を底とする対数の逆伝播)
// ============================================================================

/// 2を底とする対数の勾配関数
/// y = log2(x) の場合、∂L/∂x = ∂L/∂y / (x * ln(2))
pub struct Log2Backward<T: 'static> {
    input: Variable<T>,
    input_value: T,
}

impl<T: Clone + 'static> Log2Backward<T> {
    pub fn new(input: Variable<T>) -> Self {
        let input_value = input.value();
        Self { input, input_value }
    }
}

impl<T> GradFn<Variable<T>> for Log2Backward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Ln2
        + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // log2 の勾配: ∂L/∂x = ∂L/∂y / (x * ln(2))
        let x_ln2 = self.input_value.clone() * T::ln2();
        let grad_val = grad_y.value() / x_ln2;
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// Exp2Backward (2を底とする指数の逆伝播)
// ============================================================================

/// 2を底とする指数の勾配関数
/// y = exp2(x) = 2^x の場合、∂L/∂x = ∂L/∂y * y * ln(2)
pub struct Exp2Backward<T: 'static> {
    input: Variable<T>,
    output_value: T,
}

impl<T: Clone + 'static> Exp2Backward<T> {
    pub fn new(input: Variable<T>, output_value: T) -> Self {
        Self {
            input,
            output_value,
        }
    }
}

impl<T> GradFn<Variable<T>> for Exp2Backward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Ln2 + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // exp2 の勾配: ∂L/∂x = ∂L/∂y * y * ln(2)
        let grad_val = grad_y.value() * self.output_value.clone() * T::ln2();
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// SqrtBackward (平方根の逆伝播)
// ============================================================================

/// 平方根の勾配関数
/// y = sqrt(x) の場合、∂L/∂x = ∂L/∂y / (2 * y)
pub struct SqrtBackward<T: 'static> {
    input: Variable<T>,
    output_value: T,
}

impl<T: Clone + 'static> SqrtBackward<T> {
    pub fn new(input: Variable<T>, output_value: T) -> Self {
        Self {
            input,
            output_value,
        }
    }
}

impl<T> GradFn<Variable<T>> for SqrtBackward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Two
        + 'static,
{
    fn backward(&mut self, grad_y: Variable<T>) {
        // sqrt の勾配: ∂L/∂x = ∂L/∂y / (2 * sqrt(x)) = ∂L/∂y / (2 * y)
        let two_y = T::two() * self.output_value.clone();
        let grad_val = grad_y.value() / two_y;
        self.input.backward_with(Variable::new(grad_val));
    }
}

// ============================================================================
// Variable<T> への超越関数のブランケット実装（primops）
// ============================================================================

impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Sin + Cos + 'static,
{
    /// 正弦関数を計算
    pub fn sin(&self) -> Variable<T> {
        let output = self.value().sin();
        Variable::with_grad_fn(output, Box::new(SinBackward::new(self.clone())))
    }
}

impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + PhaseShiftQuarter + 'static,
{
    /// 位相を1/4周期シフト（π/2加算）
    pub fn phase_shift_quarter(&self) -> Variable<T> {
        let output = self.value().phase_shift_quarter();
        Variable::with_grad_fn(
            output,
            Box::new(PhaseShiftQuarterBackward::new(self.clone())),
        )
    }
}

impl<T> Variable<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Log2
        + Ln2
        + 'static,
{
    /// 2を底とする対数を計算
    pub fn log2(&self) -> Variable<T> {
        let output = self.value().log2();
        Variable::with_grad_fn(output, Box::new(Log2Backward::new(self.clone())))
    }
}

impl<T> Variable<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Exp2 + Ln2 + 'static,
{
    /// 2を底とする指数関数を計算
    pub fn exp2(&self) -> Variable<T> {
        let output = self.value().exp2();
        Variable::with_grad_fn(
            output.clone(),
            Box::new(Exp2Backward::new(self.clone(), output)),
        )
    }
}

impl<T> Variable<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Sqrt
        + Two
        + 'static,
{
    /// 平方根を計算
    pub fn sqrt(&self) -> Variable<T> {
        let output = self.value().sqrt();
        Variable::with_grad_fn(
            output.clone(),
            Box::new(SqrtBackward::new(self.clone(), output)),
        )
    }
}

// ============================================================================
// f32 / f64 へのトレイト実装
// ============================================================================

macro_rules! impl_transcendental_for_float {
    ($t:ty) => {
        impl Sin for $t {
            fn sin(&self) -> Self {
                <$t>::sin(*self)
            }
        }

        impl Cos for $t {
            fn cos(&self) -> Self {
                <$t>::cos(*self)
            }
        }

        impl Log2 for $t {
            fn log2(&self) -> Self {
                <$t>::log2(*self)
            }
        }

        impl Exp2 for $t {
            fn exp2(&self) -> Self {
                <$t>::exp2(*self)
            }
        }

        impl Sqrt for $t {
            fn sqrt(&self) -> Self {
                <$t>::sqrt(*self)
            }
        }

        impl Ln2 for $t {
            fn ln2() -> Self {
                std::f64::consts::LN_2 as $t
            }
        }

        impl Two for $t {
            fn two() -> Self {
                2.0
            }
        }

        impl PhaseShiftQuarter for $t {
            fn phase_shift_quarter(&self) -> Self {
                self + std::f64::consts::FRAC_PI_2 as $t
            }
        }
    };
}

impl_transcendental_for_float!(f32);
impl_transcendental_for_float!(f64);
