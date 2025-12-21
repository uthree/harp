//! 高級超越関数（Cos など）
//!
//! これらはprimopsの組み合わせで実装されます：
//! - Cos = PhaseShiftQuarter + Sin (cos(x) = sin(x + π/2))

use std::ops;

use crate::primops::{Cos, PhaseShiftQuarter, Sin};
use crate::variable::Variable;

// ============================================================================
// Variable<T> への Cos 実装 (hlops: PhaseShiftQuarter + Sin)
// ============================================================================

impl<T> Variable<T>
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
    pub fn cos(&self) -> Variable<T> {
        // 1/4周期シフトしてからsin適用
        self.phase_shift_quarter().sin()
    }
}
