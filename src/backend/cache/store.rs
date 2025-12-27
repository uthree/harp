//! キャッシュストア
//!
//! バックエンド共通のカーネルキャッシュ。
//! `dyn Kernel` を使用して統一的にキャッシュを管理する。

use crate::backend::KernelSignature;
use crate::backend::global::DeviceKind;
use crate::backend::pipeline::DispatchSizeConfig;
use crate::backend::traits::Kernel;
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};
use std::time::Instant;

/// キャッシュキー
///
/// グラフの文字列表現、デバイス種類、デバイス識別子で一意に識別する。
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct KernelCacheKey {
    /// グラフの文字列表現（人間可読）
    graph_repr: String,
    /// デバイス種類
    device_kind: DeviceKind,
    /// デバイス識別子（ポインタアドレス）
    /// 同じ種類でも異なるデバイスインスタンスを区別する
    device_id: usize,
}

impl KernelCacheKey {
    /// グラフ表現とデバイスからキャッシュキーを作成
    pub fn new(graph_repr: String, device_kind: DeviceKind, device_id: usize) -> Self {
        Self {
            graph_repr,
            device_kind,
            device_id,
        }
    }

    /// グラフ表現を取得（デバッグ用）
    pub fn graph_repr(&self) -> &str {
        &self.graph_repr
    }

    /// デバイス種類を取得
    pub fn device_kind(&self) -> DeviceKind {
        self.device_kind
    }

    /// デバイス識別子を取得
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

impl std::fmt::Debug for KernelCacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CacheKey({:?}, id={:#x}, {})",
            self.device_kind, self.device_id, &self.graph_repr
        )
    }
}

/// 統一キャッシュエントリ
///
/// バックエンドに依存しない統一的なキャッシュエントリ。
/// `Box<dyn Kernel>` を使用してカーネルを格納する。
pub struct CacheEntry {
    /// コンパイル済みカーネル
    pub kernel: Box<dyn Kernel>,
    /// カーネル署名
    pub signature: KernelSignature,
    /// ディスパッチサイズ設定
    pub dispatch_config: DispatchSizeConfig,
    /// 最終アクセス時刻
    pub last_accessed: Instant,
}

impl Clone for CacheEntry {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone_kernel(),
            signature: self.signature.clone(),
            dispatch_config: self.dispatch_config.clone(),
            last_accessed: self.last_accessed,
        }
    }
}

impl CacheEntry {
    /// 新しいキャッシュエントリを作成
    pub fn new(
        kernel: Box<dyn Kernel>,
        signature: KernelSignature,
        dispatch_config: DispatchSizeConfig,
    ) -> Self {
        Self {
            kernel,
            signature,
            dispatch_config,
            last_accessed: Instant::now(),
        }
    }
}

/// キャッシュの統計情報
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub entries: usize,
    pub evictions: usize,
}

/// グローバルキャッシュの最大エントリ数
const MAX_CACHE_ENTRIES: usize = 1024;

/// グローバルキャッシュ
static KERNEL_CACHE: LazyLock<RwLock<HashMap<KernelCacheKey, CacheEntry>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// キャッシュ統計
static CACHE_STATS: LazyLock<RwLock<CacheStats>> =
    LazyLock::new(|| RwLock::new(CacheStats::default()));

/// キャッシュからカーネルを取得
pub fn get_cached_kernel(key: &KernelCacheKey) -> Option<CacheEntry> {
    let cache = KERNEL_CACHE.read().unwrap();
    if let Some(entry) = cache.get(key) {
        let mut stats = CACHE_STATS.write().unwrap();
        stats.hits += 1;

        Some(CacheEntry {
            kernel: entry.kernel.clone_kernel(),
            signature: entry.signature.clone(),
            dispatch_config: entry.dispatch_config.clone(),
            last_accessed: Instant::now(),
        })
    } else {
        let mut stats = CACHE_STATS.write().unwrap();
        stats.misses += 1;
        None
    }
}

/// キャッシュにカーネルを挿入
pub fn insert_cached_kernel(key: KernelCacheKey, entry: CacheEntry) {
    let mut cache = KERNEL_CACHE.write().unwrap();

    // LRU eviction
    if cache.len() >= MAX_CACHE_ENTRIES {
        let oldest_key = cache
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(k) = oldest_key {
            cache.remove(&k);
            let mut stats = CACHE_STATS.write().unwrap();
            stats.evictions += 1;
        }
    }

    cache.insert(key, entry);
    let mut stats = CACHE_STATS.write().unwrap();
    stats.entries = cache.len();
}

/// キャッシュの統計情報を取得
pub fn get_cache_stats() -> CacheStats {
    CACHE_STATS.read().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_equality() {
        // 同じグラフ表現、デバイス、デバイスIDのキーは等しい
        let key1 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::Metal, 0x1000);
        let key2 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::Metal, 0x1000);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_graphs() {
        // 異なるグラフ表現は異なるキー
        let key1 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::Metal, 0x1000);
        let key2 = KernelCacheKey::new("Mul($0, $1)".to_string(), DeviceKind::Metal, 0x1000);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_devices() {
        // 異なるデバイス種類は異なるキー
        let key1 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::Metal, 0x1000);
        let key2 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::OpenCL, 0x1000);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_device_ids() {
        // 同じ種類でも異なるデバイスIDは異なるキー
        let key1 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::OpenCL, 0x1000);
        let key2 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::OpenCL, 0x2000);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_hash() {
        use std::collections::HashMap;

        // HashMapで使えることを確認
        let mut map: HashMap<KernelCacheKey, i32> = HashMap::new();
        let key = KernelCacheKey::new("test".to_string(), DeviceKind::Metal, 0x1000);
        map.insert(key.clone(), 42);
        assert_eq!(map.get(&key), Some(&42));
    }

    #[test]
    fn test_cache_key_getters() {
        let key = KernelCacheKey::new("Graph(...)".to_string(), DeviceKind::OpenCL, 0xABCD);
        assert_eq!(key.graph_repr(), "Graph(...)");
        assert_eq!(key.device_kind(), DeviceKind::OpenCL);
        assert_eq!(key.device_id(), 0xABCD);
    }

    #[test]
    fn test_cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.evictions, 0);
    }
}
