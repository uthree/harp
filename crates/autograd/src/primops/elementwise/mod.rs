//! 要素単位演算 (Elementwise Operations)
//!
//! スカラー・テンソル両方で使用可能な要素単位の演算を定義します。
//! - 四則演算（加減乗除、剰余）
//! - 超越関数（sin, cos, exp, ln など）
//! - 型変換

pub mod arithmetic;
pub mod cast;
pub mod transcendental;

// 四則演算
pub use arithmetic::{
    AddBackward, Floor, Maximum, MaximumBackward, MulBackward, NegBackward, RecipBackward,
    RemBackward,
};

// 型変換
pub use cast::CastBackward;

// 超越関数
pub use transcendental::{
    Cos, Exp2, Exp2Backward, Ln2, Log2, Log2Backward, Log2E, MulLn2, MulLn2Backward, MulLog2E,
    MulLog2EBackward, PhaseShiftQuarter, PhaseShiftQuarterBackward, Sin, SinBackward, Sqrt,
    SqrtBackward, Two,
};
