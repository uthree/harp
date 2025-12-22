//! プリミティブ演算 (primops)
//!
//! 直接実装され、独自の逆伝播を持つ基本演算を定義します。
//! 高級演算(hlops)はこれらの演算の組み合わせで実装されます。
//!
//! ## モジュール構成
//!
//! - `arithmetic`: 四則演算（加減乗除、剰余、maximum）
//! - `transcendental`: 超越関数（sin, cos, exp2, log2, sqrt）
//! - `array`: 配列演算（縮約・拡張、次元操作、線形代数）

pub mod arithmetic;
pub mod transcendental;

// 四則演算の再エクスポート
pub use arithmetic::{
    // 四則演算
    AddBackward,
    Floor,
    Maximum,
    MaximumBackward,
    MulBackward,
    NegBackward,
    RecipBackward,
    RemBackward,
    // 型変換
    cast::CastBackward,
};

// 超越関数の再エクスポート（公開API）
pub use transcendental::{
    Exp2, Exp2Backward, Log2, Log2Backward, Log2E, Pi, Sin, SinBackward, Sqrt, SqrtBackward,
};
