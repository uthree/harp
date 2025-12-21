//! 高級演算 (hlops)
//!
//! プリミティブ演算(primops)の組み合わせで実装される演算です。
//! ブランケット実装によりprimopsにフォールバックします。

pub mod arithmetic;
pub mod linalg;
pub mod transcendental;

// 現在、hlopsは演算子トレイト実装として提供されるため、
// 個別の型をエクスポートする必要はありません。
// 将来的に独自の構造体が必要になった場合はここに追加します。
