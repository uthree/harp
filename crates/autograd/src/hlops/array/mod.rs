//! 配列の高級演算 (Array HLOPs)
//!
//! primops の組み合わせで実装される配列演算:
//! - Matmul = Unsqueeze + Expand + Mul + Sum + Squeeze (汎用フォールバック)

pub mod linalg;

pub use linalg::matmul_fallback;
