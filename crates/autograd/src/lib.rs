//! 自動微分ライブラリ
//!
//! このクレートは計算グラフベースの自動微分を提供します。
//!
//! ## アーキテクチャ
//!
//! - **primops**: プリミティブ演算（直接実装、独自の逆伝播を持つ）
//! - **hlops**: 高級演算（primopsの組み合わせで実装）

pub mod hlops;
#[cfg(feature = "ndarray")]
mod ndarray_impl;
pub mod primops;
mod traits;
pub mod variable;

// primops からのエクスポート
pub use primops::{
    // 四則演算
    AddBackward,
    // 型変換
    CastBackward,
    // 超越関数
    Cos,
    Exp2,
    Exp2Backward,
    // 縮約・拡張演算
    Expand,
    ExpandBackward,
    Floor,
    Ln2,
    Log2,
    Log2Backward,
    Log2E,
    Max,
    MaxBackward,
    MulBackward,
    MulLn2,
    MulLn2Backward,
    MulLog2E,
    MulLog2EBackward,
    NegBackward,
    PhaseShiftQuarter,
    PhaseShiftQuarterBackward,
    Prod,
    ProdBackward,
    RecipBackward,
    RemBackward,
    RemOp,
    Sin,
    SinBackward,
    Sqrt,
    SqrtBackward,
    Sum,
    SumBackward,
    Two,
};

// hlops からのエクスポート
pub use hlops::arithmetic::One;

// traits からのエクスポート
pub use traits::{GradFn, GradNode, GradRoot};

// variable からのエクスポート
pub use variable::Variable;
