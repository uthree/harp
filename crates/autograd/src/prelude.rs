//! よく使われる型・トレイトを一括インポートするためのモジュール
//!
//! # 使用例
//!
//! ```rust
//! use harp_autograd::prelude::*;
//!
//! let x = Differentiable::new(1.0_f32);
//! let y = x.sin();
//! y.backward();
//! ```

// ============================================================================
// Core
// ============================================================================

pub use crate::Differentiable;
pub use crate::shape::IntoShape;
pub use crate::traits::{Arithmetic, Array, GradNode, GradRoot, Transcendental};

// ============================================================================
// 要素単位演算（トレイト）
// ============================================================================

pub use crate::primops::{
    // 超越関数
    Exp2,
    Floor,
    Log2,
    Maximum,
    Sin,
    Sqrt,
};
