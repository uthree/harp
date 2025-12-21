//! 要素単位の高級演算 (Elementwise HLOPs)
//!
//! primops の組み合わせで実装される要素単位演算：
//! - Sub = Add + Neg
//! - Div = Mul + Recip
//! - Cos = PhaseShiftQuarter + Sin
//! - Ln = Log2 + MulLn2
//! - Exp = MulLog2E + Exp2

pub mod arithmetic;
pub mod transcendental;

pub use arithmetic::One;
