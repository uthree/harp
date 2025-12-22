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
pub mod array;
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
    Cos, Exp2, Exp2Backward, Log2, Log2Backward, Sin, SinBackward, Sqrt, SqrtBackward,
};

// 超越関数の再エクスポート（内部実装詳細、hlops で使用）
pub(crate) use transcendental::{Ln2, Log2E, MulLn2, MulLog2E, PhaseShiftQuarter};

// 配列演算の再エクスポート
pub use array::{
    // 縮約・拡張演算
    Expand,
    ExpandBackward,
    // 線形代数
    Matmul,
    MatmulBackward,
    Max,
    MaxBackward,
    // 形状情報
    Ndim,
    // 初期化
    Ones,
    // 軸順序変更
    Permute,
    PermuteBackward,
    Prod,
    ProdBackward,
    Rand,
    // 形状変更
    Reshape,
    ReshapeBackward,
    Shape,
    // 次元操作
    Squeeze,
    SqueezeBackward,
    Sum,
    SumBackward,
    Unsqueeze,
    UnsqueezeBackward,
    Zeros,
    inverse_permutation,
};
