//! カーネルキャッシュ
//!
//! グラフ単位でコンパイル済みカーネルをキャッシュし、
//! 同一構造のグラフに対してGPUコンパイルをスキップする。
//!
//! - `store`: メモリ内キャッシュ
//! - `disk`: ディスクキャッシュ（セッション間で永続化）

pub mod disk;
mod store;

pub use store::{
    CacheEntry, CacheStats, KernelCacheKey, get_cache_stats, get_cached_kernel,
    insert_cached_kernel,
};

use crate::ast::AstNode;
use crate::backend::global::DeviceKind;

/// ASTからキャッシュキーを生成
///
/// ASTのDebug実装を使用して一意な文字列表現を生成し、
/// デバイス情報と組み合わせてキャッシュキーを作成する。
pub fn generate_cache_key(
    kernel_ast: &AstNode,
    device_kind: DeviceKind,
    device_id: usize,
) -> KernelCacheKey {
    let graph_repr = format!("{:?}", kernel_ast);
    KernelCacheKey::new(graph_repr, device_kind, device_id)
}
