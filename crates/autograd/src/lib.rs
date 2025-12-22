//! 自動微分ライブラリ
//!
//! このクレートは計算グラフベースの自動微分を提供します。
//!
//! ## 使用方法
//!
//! ```rust
//! use autograd::prelude::*;
//!
//! let x = Variable::new(1.0_f32);
//! let y = x.sin();
//! y.backward();
//! ```
//!
//! ## アーキテクチャ
//!
//! - **primops**: プリミティブ演算（直接実装、独自の逆伝播を持つ）
//! - **hlops**: 高級演算（primopsの組み合わせで実装）

pub mod differentiable;
pub mod hlops;
#[cfg(feature = "ndarray")]
mod ndarray_impl;
pub mod prelude;
pub mod primops;
pub mod shape;
pub mod traits;

// ============================================================================
// 主要な型・トレイト（トップレベルからもアクセス可能）
// ============================================================================

// Core
pub use differentiable::Differentiable;
pub use shape::IntoShape;
pub use traits::{Arithmetic, Array, GradFn, GradNode, GradRoot, Transcendental};

// 演算トレイト
pub use primops::{
    // 要素単位
    Cos,
    Exp2,
    // 構造変更
    Expand,
    Floor,
    Log2,
    Matmul,
    Max,
    Maximum,
    Ndim,
    // 初期化
    Ones,
    Permute,
    Prod,
    Reshape,
    Shape,
    Sin,
    Sqrt,
    Squeeze,
    Sum,
    Unsqueeze,
    Zeros,
};

// 定数
pub use hlops::One;

// ユーティリティ
pub use primops::inverse_permutation;

// ============================================================================
// Backward 構造体（内部実装詳細、ドキュメントから隠す）
// ============================================================================

#[doc(hidden)]
pub use primops::{
    AddBackward, CastBackward, Exp2Backward, ExpandBackward, Log2Backward, MatmulBackward,
    MaxBackward, MaximumBackward, MulBackward, NegBackward, PermuteBackward, ProdBackward,
    RecipBackward, RemBackward, ReshapeBackward, SinBackward, SqrtBackward, SqueezeBackward,
    SumBackward, UnsqueezeBackward,
};
