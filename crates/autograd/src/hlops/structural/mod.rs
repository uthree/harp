//! 構造変更の高級演算 (Structural HLOPs)
//!
//! primops の組み合わせで実装される構造変更演算：
//! - Matmul = Unsqueeze + Expand + Mul + Sum + Squeeze (汎用フォールバック)

pub mod linalg;

pub use linalg::matmul_fallback;
