//! プリミティブ演算 (primops)
//!
//! 直接実装され、独自の逆伝播を持つ基本演算を定義します。
//! 高級演算(hlops)はこれらの演算の組み合わせで実装されます。

pub mod arithmetic;
pub mod cast;
pub mod reduce;
pub mod transcendental;

// 四則演算
pub use arithmetic::{
    AddBackward, Floor, MulBackward, NegBackward, RecipBackward, RemBackward, RemOp,
};

// 型変換
pub use cast::CastBackward;

// 縮約・拡張演算
pub use reduce::{Expand, ExpandBackward, Max, MaxBackward, Prod, ProdBackward, Sum, SumBackward};

// 超越関数
pub use transcendental::{
    Cos, Exp2, Exp2Backward, Ln2, Log2, Log2Backward, PhaseShiftQuarter, PhaseShiftQuarterBackward,
    Sin, SinBackward, Sqrt, SqrtBackward, Two,
};
