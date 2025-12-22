//! 高級超越関数（Cos, Ln, Exp など）
//!
//! これらはprimopsの組み合わせで実装されます：
//! - Cos = PhaseShiftQuarter + Sin (cos(x) = sin(x + π/2))
//! - Ln = Log2 + MulLn2 (ln(x) = log2(x) * ln(2))
//! - Exp = MulLog2E + Exp2 (exp(x) = exp2(x * log2(e)))

use std::ops;

use crate::differentiable::Differentiable;
use crate::primops::{Cos, Exp2, Ln2, Log2, Log2E, MulLn2, MulLog2E, PhaseShiftQuarter, Sin};

// ============================================================================
// Variable<T> への Cos 実装 (hlops: PhaseShiftQuarter + Sin)
// ============================================================================

impl<T> Differentiable<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + Sin
        + Cos
        + PhaseShiftQuarter
        + 'static,
{
    /// 余弦関数を計算
    /// cos(x) = sin(x + π/2) として実装
    pub fn cos(&self) -> Differentiable<T> {
        // 1/4周期シフトしてからsin適用
        self.phase_shift_quarter().sin()
    }
}

// ============================================================================
// Variable<T> への Ln 実装 (hlops: Log2 + MulLn2)
// ============================================================================

impl<T> Differentiable<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + Log2
        + Ln2
        + MulLn2
        + 'static,
{
    /// 自然対数を計算
    /// ln(x) = log2(x) * ln(2) として実装
    pub fn ln(&self) -> Differentiable<T> {
        self.log2().mul_ln2()
    }
}

// ============================================================================
// Variable<T> への Exp 実装 (hlops: MulLog2E + Exp2)
// ============================================================================

impl<T> Differentiable<T>
where
    T: Clone
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + Exp2
        + Ln2
        + MulLog2E
        + Log2E
        + 'static,
{
    /// 自然指数関数を計算
    /// exp(x) = exp2(x * log2(e)) として実装
    pub fn exp(&self) -> Differentiable<T> {
        self.mul_log2e().exp2()
    }
}
