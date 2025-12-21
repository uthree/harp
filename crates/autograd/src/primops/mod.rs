//! プリミティブ演算 (primops)
//!
//! 直接実装され、独自の逆伝播を持つ基本演算を定義します。
//! 高級演算(hlops)はこれらの演算の組み合わせで実装されます。
//!
//! ## モジュール構成
//!
//! - `elementwise`: 要素単位演算（スカラー・テンソル両対応）
//!   - 四則演算、超越関数、型変換
//! - `structural`: 構造変更演算（テンソル専用）
//!   - 縮約・拡張、線形代数、形状情報

pub mod elementwise;
pub mod structural;

// 要素単位演算の再エクスポート
pub use elementwise::{
    // 四則演算
    AddBackward,
    // 型変換
    CastBackward,
    // 超越関数
    Cos,
    Exp2,
    Exp2Backward,
    Floor,
    Ln2,
    Log2,
    Log2Backward,
    Log2E,
    Maximum,
    MaximumBackward,
    MulBackward,
    MulLn2,
    MulLn2Backward,
    MulLog2E,
    MulLog2EBackward,
    NegBackward,
    PhaseShiftQuarter,
    PhaseShiftQuarterBackward,
    RecipBackward,
    RemBackward,
    Sin,
    SinBackward,
    Sqrt,
    SqrtBackward,
    Two,
};

// 構造変更演算の再エクスポート
pub use structural::{
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
    // 軸順序変更
    Permute,
    PermuteBackward,
    Prod,
    ProdBackward,
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
    inverse_permutation,
};
