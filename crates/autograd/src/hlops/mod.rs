//! 高級演算 (hlops)
//!
//! プリミティブ演算(primops)の組み合わせで実装される演算です。
//! ブランケット実装によりprimopsにフォールバックします。
//!
//! ## モジュール構成
//!
//! - `arithmetic`: 四則演算（Sub, Div）
//! - `transcendental`: 超越関数（Cos, Ln, Exp）
//! - `array`: 配列演算（matmul_fallback）

pub mod arithmetic;
pub mod transcendental;
