//! 超越関数（Sin, Exp2, Log2, Sqrt など）のプリミティブ演算

use std::ops;

use crate::differentiable::Differentiable;
use crate::traits::GradFn;

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

/// 自身に ln(2) を乗算するトレイト
/// ln(x) = log2(x) * ln(2) を実現するための基本演算
pub trait MulLn2 {
    fn mul_ln2(&self) -> Self;
}

/// 自身に log2(e) = 1/ln(2) を乗算するトレイト
/// exp(x) = exp2(x * log2(e)) を実現するための基本演算
pub trait MulLog2E {
    fn mul_log2e(&self) -> Self;
}

// ============================================================================
// SinBackward (正弦の逆伝播)
// ============================================================================

/// 正弦の勾配関数
/// y = sin(x) の場合、∂L/∂x = ∂L/∂y * cos(x)
pub struct SinBackward<T: 'static> {
    input: Differentiable<T>,
    input_value: T,
}

impl<T: Clone + 'static> SinBackward<T> {
    pub fn new(input: Differentiable<T>) -> Self {
        let input_value = input.value();
        Self { input, input_value }
    }
}

impl<T> GradFn<Differentiable<T>> for SinBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Cos + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // sin の勾配: ∂L/∂x = ∂L/∂y * cos(x)
        let requires_grad = grad_y.requires_grad();
        let cos_x = self.input_value.cos();
        let grad_val = grad_y.value() * cos_x;
        self.input
            .backward_with(Differentiable::new_with_requires_grad(
                grad_val,
                requires_grad,
            ));
    }
}

// ============================================================================
// PhaseShiftQuarterBackward (1/4周期シフトの逆伝播)
// ============================================================================

/// 1/4周期シフトの勾配関数（定数加算なので勾配はそのまま通過）
/// y = x + π/2 の場合、∂L/∂x = ∂L/∂y
pub struct PhaseShiftQuarterBackward<T: 'static> {
    input: Differentiable<T>,
}

impl<T: 'static> PhaseShiftQuarterBackward<T> {
    pub fn new(input: Differentiable<T>) -> Self {
        Self { input }
    }
}

impl<T> GradFn<Differentiable<T>> for PhaseShiftQuarterBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // 定数加算の勾配: そのまま通過
        self.input.backward_with(grad_y);
    }
}

// ============================================================================
// MulLn2Backward (ln(2)乗算の逆伝播)
// ============================================================================

/// ln(2)乗算の勾配関数
/// y = x * ln(2) の場合、∂L/∂x = ∂L/∂y * ln(2)
pub struct MulLn2Backward<T: 'static> {
    input: Differentiable<T>,
}

impl<T: 'static> MulLn2Backward<T> {
    pub fn new(input: Differentiable<T>) -> Self {
        Self { input }
    }
}

impl<T> GradFn<Differentiable<T>> for MulLn2Backward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Ln2 + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // ln(2)乗算の勾配: ∂L/∂x = ∂L/∂y * ln(2)
        let requires_grad = grad_y.requires_grad();
        let grad_val = grad_y.value() * T::ln2();
        self.input
            .backward_with(Differentiable::new_with_requires_grad(
                grad_val,
                requires_grad,
            ));
    }
}

// ============================================================================
// MulLog2EBackward (log2(e)乗算の逆伝播)
// ============================================================================

/// log2(e) を取得するトレイト
pub trait Log2E {
    fn log2e() -> Self;
}

/// log2(e)乗算の勾配関数
/// y = x * log2(e) の場合、∂L/∂x = ∂L/∂y * log2(e)
pub struct MulLog2EBackward<T: 'static> {
    input: Differentiable<T>,
}

impl<T: 'static> MulLog2EBackward<T> {
    pub fn new(input: Differentiable<T>) -> Self {
        Self { input }
    }
}

impl<T> GradFn<Differentiable<T>> for MulLog2EBackward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Log2E + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // log2(e)乗算の勾配: ∂L/∂x = ∂L/∂y * log2(e)
        let requires_grad = grad_y.requires_grad();
        let grad_val = grad_y.value() * T::log2e();
        self.input
            .backward_with(Differentiable::new_with_requires_grad(
                grad_val,
                requires_grad,
            ));
    }
}

// ============================================================================
// Log2Backward (2を底とする対数の逆伝播)
// ============================================================================

/// 2を底とする対数の勾配関数
/// y = log2(x) の場合、∂L/∂x = ∂L/∂y / (x * ln(2))
pub struct Log2Backward<T: 'static> {
    input: Differentiable<T>,
    input_value: T,
}

impl<T: Clone + 'static> Log2Backward<T> {
    pub fn new(input: Differentiable<T>) -> Self {
        let input_value = input.value();
        Self { input, input_value }
    }
}

impl<T> GradFn<Differentiable<T>> for Log2Backward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Ln2
        + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // log2 の勾配: ∂L/∂x = ∂L/∂y / (x * ln(2))
        let requires_grad = grad_y.requires_grad();
        let x_ln2 = self.input_value.clone() * T::ln2();
        let grad_val = grad_y.value() / x_ln2;
        self.input
            .backward_with(Differentiable::new_with_requires_grad(
                grad_val,
                requires_grad,
            ));
    }
}

// ============================================================================
// Exp2Backward (2を底とする指数の逆伝播)
// ============================================================================

/// 2を底とする指数の勾配関数
/// y = exp2(x) = 2^x の場合、∂L/∂x = ∂L/∂y * y * ln(2)
pub struct Exp2Backward<T: 'static> {
    input: Differentiable<T>,
    output_value: T,
}

impl<T: Clone + 'static> Exp2Backward<T> {
    pub fn new(input: Differentiable<T>, output_value: T) -> Self {
        Self {
            input,
            output_value,
        }
    }
}

impl<T> GradFn<Differentiable<T>> for Exp2Backward<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Ln2 + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // exp2 の勾配: ∂L/∂x = ∂L/∂y * y * ln(2)
        let requires_grad = grad_y.requires_grad();
        let grad_val = grad_y.value() * self.output_value.clone() * T::ln2();
        self.input
            .backward_with(Differentiable::new_with_requires_grad(
                grad_val,
                requires_grad,
            ));
    }
}

// ============================================================================
// SqrtBackward (平方根の逆伝播)
// ============================================================================

/// 平方根の勾配関数
/// y = sqrt(x) の場合、∂L/∂x = ∂L/∂y / (2 * y)
pub struct SqrtBackward<T: 'static> {
    input: Differentiable<T>,
    output_value: T,
}

impl<T: Clone + 'static> SqrtBackward<T> {
    pub fn new(input: Differentiable<T>, output_value: T) -> Self {
        Self {
            input,
            output_value,
        }
    }
}

impl<T> GradFn<Differentiable<T>> for SqrtBackward<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Two
        + 'static,
{
    fn backward(&mut self, grad_y: Differentiable<T>) {
        // sqrt の勾配: ∂L/∂x = ∂L/∂y / (2 * sqrt(x)) = ∂L/∂y / (2 * y)
        let requires_grad = grad_y.requires_grad();
        let two_y = T::two() * self.output_value.clone();
        let grad_val = grad_y.value() / two_y;
        self.input
            .backward_with(Differentiable::new_with_requires_grad(
                grad_val,
                requires_grad,
            ));
    }
}

// ============================================================================
// Variable<T> への超越関数のブランケット実装（primops）
// ============================================================================

impl<T> Differentiable<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Sin + Cos + 'static,
{
    /// 正弦関数を計算
    pub fn sin(&self) -> Differentiable<T> {
        let output = self.value().sin();
        if self.requires_grad() {
            Differentiable::with_grad_fn(output, Box::new(SinBackward::new(self.clone())))
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}

impl<T> Differentiable<T>
where
    T: Clone + ops::Add<T, Output = T> + PhaseShiftQuarter + 'static,
{
    /// 位相を1/4周期シフト（π/2加算）
    pub fn phase_shift_quarter(&self) -> Differentiable<T> {
        let output = self.value().phase_shift_quarter();
        if self.requires_grad() {
            Differentiable::with_grad_fn(
                output,
                Box::new(PhaseShiftQuarterBackward::new(self.clone())),
            )
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}

impl<T> Differentiable<T>
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
    pub fn log2(&self) -> Differentiable<T> {
        let output = self.value().log2();
        if self.requires_grad() {
            Differentiable::with_grad_fn(output, Box::new(Log2Backward::new(self.clone())))
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}

impl<T> Differentiable<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + Exp2 + Ln2 + 'static,
{
    /// 2を底とする指数関数を計算
    pub fn exp2(&self) -> Differentiable<T> {
        let output = self.value().exp2();
        if self.requires_grad() {
            Differentiable::with_grad_fn(
                output.clone(),
                Box::new(Exp2Backward::new(self.clone(), output)),
            )
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}

impl<T> Differentiable<T>
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
    pub fn sqrt(&self) -> Differentiable<T> {
        let output = self.value().sqrt();
        if self.requires_grad() {
            Differentiable::with_grad_fn(
                output.clone(),
                Box::new(SqrtBackward::new(self.clone(), output)),
            )
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}

impl<T> Differentiable<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + MulLn2 + Ln2 + 'static,
{
    /// 自身に ln(2) を乗算
    pub fn mul_ln2(&self) -> Differentiable<T> {
        let output = self.value().mul_ln2();
        if self.requires_grad() {
            Differentiable::with_grad_fn(output, Box::new(MulLn2Backward::new(self.clone())))
        } else {
            Differentiable::new_no_grad(output)
        }
    }
}

impl<T> Differentiable<T>
where
    T: Clone + ops::Add<T, Output = T> + ops::Mul<T, Output = T> + MulLog2E + Log2E + 'static,
{
    /// 自身に log2(e) を乗算
    pub fn mul_log2e(&self) -> Differentiable<T> {
        let output = self.value().mul_log2e();
        if self.requires_grad() {
            Differentiable::with_grad_fn(output, Box::new(MulLog2EBackward::new(self.clone())))
        } else {
            Differentiable::new_no_grad(output)
        }
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

        impl MulLn2 for $t {
            fn mul_ln2(&self) -> Self {
                self * std::f64::consts::LN_2 as $t
            }
        }

        impl MulLog2E for $t {
            fn mul_log2e(&self) -> Self {
                self * std::f64::consts::LOG2_E as $t
            }
        }

        impl Log2E for $t {
            fn log2e() -> Self {
                std::f64::consts::LOG2_E as $t
            }
        }
    };
}

impl_transcendental_for_float!(f32);
impl_transcendental_for_float!(f64);
