//! 配列の高級演算 (Array HLOPs)
//!
//! primops の組み合わせで実装される配列演算:
//! - Matmul = Unsqueeze + Expand + Mul + Sum + Squeeze (汎用フォールバック)
//! - Randn = Box-Muller変換 (Rand + Log2 + Sqrt + Cos)

pub mod initialization;
pub mod linalg;

pub use initialization::{BoxMullerConstants, RandnDefault};
pub use linalg::matmul_fallback;
