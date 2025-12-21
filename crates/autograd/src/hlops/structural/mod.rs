//! 構造変更の高級演算 (Structural HLOPs)
//!
//! primops の組み合わせで実装される構造変更演算：
//! - Matmul = Expand + Mul + Sum (汎用フォールバック)

pub mod linalg;
