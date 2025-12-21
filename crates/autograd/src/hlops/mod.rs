//! 高級演算 (hlops)
//!
//! プリミティブ演算(primops)の組み合わせで実装される演算です。
//! ブランケット実装によりprimopsにフォールバックします。
//!
//! ## モジュール構成
//!
//! - `elementwise`: 要素単位演算
//!   - Sub, Div, Cos, Ln, Exp
//! - `structural`: 構造変更演算
//!   - matmul_fallback (汎用フォールバック)

pub mod elementwise;
pub mod structural;

pub use elementwise::One;
pub use structural::matmul_fallback;
