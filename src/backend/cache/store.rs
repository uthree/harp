//! キャッシュストア
//!
//! バックエンド別にコンパイル済みカーネルをキャッシュする。

use crate::backend::KernelSignature;
use crate::backend::global::DeviceKind;
use crate::backend::pipeline::DispatchSizeConfig;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backend::metal::MetalKernel;

#[cfg(feature = "opencl")]
use crate::backend::opencl::OpenCLKernel;

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

/// キャッシュエントリの共通情報
pub trait CachedKernelEntry {
    /// カーネル署名を取得
    fn signature(&self) -> &KernelSignature;
    /// ディスパッチ設定を取得
    fn dispatch_config(&self) -> &DispatchSizeConfig;
}

/// Metalカーネルのキャッシュエントリ
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct MetalCacheEntry {
    pub kernel: MetalKernel,
    pub signature: KernelSignature,
    pub dispatch_config: DispatchSizeConfig,
    pub last_accessed: std::time::Instant,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl CachedKernelEntry for MetalCacheEntry {
    fn signature(&self) -> &KernelSignature {
        &self.signature
    }

    fn dispatch_config(&self) -> &DispatchSizeConfig {
        &self.dispatch_config
    }
}

/// OpenCLカーネルのキャッシュエントリ
#[cfg(feature = "opencl")]
pub struct OpenCLCacheEntry {
    pub kernel: OpenCLKernel,
    pub signature: KernelSignature,
    pub dispatch_config: DispatchSizeConfig,
    pub last_accessed: std::time::Instant,
}

#[cfg(feature = "opencl")]
impl CachedKernelEntry for OpenCLCacheEntry {
    fn signature(&self) -> &KernelSignature {
        &self.signature
    }

    fn dispatch_config(&self) -> &DispatchSizeConfig {
        &self.dispatch_config
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
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
const MAX_CACHE_ENTRIES: usize = 1024;

// ============================================================================
// Metalキャッシュ
// ============================================================================

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_cache {
    use super::{CacheStats, KernelCacheKey, MAX_CACHE_ENTRIES, MetalCacheEntry};
    use std::collections::HashMap;
    use std::sync::{LazyLock, RwLock};
    use std::time::Instant;

    static METAL_CACHE: LazyLock<RwLock<HashMap<KernelCacheKey, MetalCacheEntry>>> =
        LazyLock::new(|| RwLock::new(HashMap::new()));
    static METAL_STATS: LazyLock<RwLock<CacheStats>> =
        LazyLock::new(|| RwLock::new(CacheStats::default()));

    /// Metalキャッシュからカーネルを取得
    pub fn get_metal_kernel(key: &KernelCacheKey) -> Option<MetalCacheEntry> {
        let cache = METAL_CACHE.read().unwrap();
        if let Some(entry) = cache.get(key) {
            // 統計更新
            let mut stats = METAL_STATS.write().unwrap();
            stats.hits += 1;

            // エントリをクローン（last_accessedは更新しない - 読み取り専用）
            Some(MetalCacheEntry {
                kernel: entry.kernel.clone(),
                signature: entry.signature.clone(),
                dispatch_config: entry.dispatch_config.clone(),
                last_accessed: Instant::now(),
            })
        } else {
            let mut stats = METAL_STATS.write().unwrap();
            stats.misses += 1;
            None
        }
    }

    /// Metalキャッシュにカーネルを挿入
    pub fn insert_metal_kernel(key: KernelCacheKey, entry: MetalCacheEntry) {
        let mut cache = METAL_CACHE.write().unwrap();

        // LRU eviction
        if cache.len() >= MAX_CACHE_ENTRIES {
            // 最も古いエントリを削除
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, e): &(&KernelCacheKey, &MetalCacheEntry)| e.last_accessed)
                .map(|(k, _): (&KernelCacheKey, &MetalCacheEntry)| k.clone());

            if let Some(k) = oldest_key {
                cache.remove(&k);
                let mut stats = METAL_STATS.write().unwrap();
                stats.evictions += 1;
            }
        }

        cache.insert(key, entry);
        let mut stats = METAL_STATS.write().unwrap();
        stats.entries = cache.len();
    }

    /// Metal キャッシュの統計を取得
    pub fn metal_cache_stats() -> CacheStats {
        METAL_STATS.read().unwrap().clone()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_cache::{get_metal_kernel, insert_metal_kernel, metal_cache_stats};

// ============================================================================
// OpenCLキャッシュ
// ============================================================================

#[cfg(feature = "opencl")]
mod opencl_cache {
    use super::{CacheStats, KernelCacheKey, MAX_CACHE_ENTRIES, OpenCLCacheEntry};
    use std::collections::HashMap;
    use std::sync::{LazyLock, RwLock};
    use std::time::Instant;

    static OPENCL_CACHE: LazyLock<RwLock<HashMap<KernelCacheKey, OpenCLCacheEntry>>> =
        LazyLock::new(|| RwLock::new(HashMap::new()));
    static OPENCL_STATS: LazyLock<RwLock<CacheStats>> =
        LazyLock::new(|| RwLock::new(CacheStats::default()));

    /// OpenCLキャッシュからカーネルを取得
    pub fn get_opencl_kernel(key: &KernelCacheKey) -> Option<OpenCLCacheEntry> {
        let cache = OPENCL_CACHE.read().unwrap();
        if let Some(entry) = cache.get(key) {
            let mut stats = OPENCL_STATS.write().unwrap();
            stats.hits += 1;

            Some(OpenCLCacheEntry {
                kernel: entry.kernel.clone(),
                signature: entry.signature.clone(),
                dispatch_config: entry.dispatch_config.clone(),
                last_accessed: Instant::now(),
            })
        } else {
            let mut stats = OPENCL_STATS.write().unwrap();
            stats.misses += 1;
            None
        }
    }

    /// OpenCLキャッシュにカーネルを挿入
    pub fn insert_opencl_kernel(key: KernelCacheKey, entry: OpenCLCacheEntry) {
        let mut cache = OPENCL_CACHE.write().unwrap();

        // LRU eviction
        if cache.len() >= MAX_CACHE_ENTRIES {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, e): &(&KernelCacheKey, &OpenCLCacheEntry)| e.last_accessed)
                .map(|(k, _): (&KernelCacheKey, &OpenCLCacheEntry)| k.clone());

            if let Some(k) = oldest_key {
                cache.remove(&k);
                let mut stats = OPENCL_STATS.write().unwrap();
                stats.evictions += 1;
            }
        }

        cache.insert(key, entry);
        let mut stats = OPENCL_STATS.write().unwrap();
        stats.entries = cache.len();
    }

    /// OpenCL キャッシュの統計を取得
    pub fn opencl_cache_stats() -> CacheStats {
        OPENCL_STATS.read().unwrap().clone()
    }
}

#[cfg(feature = "opencl")]
pub use opencl_cache::{get_opencl_kernel, insert_opencl_kernel, opencl_cache_stats};

// ============================================================================
// 汎用インターフェース
// ============================================================================

/// キャッシュからカーネルを取得（デバイス種類に応じて適切なキャッシュを使用）
pub fn get_cached_kernel(key: &KernelCacheKey) -> bool {
    match key.device_kind() {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        DeviceKind::Metal => get_metal_kernel(key).is_some(),
        #[cfg(feature = "opencl")]
        DeviceKind::OpenCL => get_opencl_kernel(key).is_some(),
        _ => false,
    }
}

/// キャッシュにカーネルを挿入（プレースホルダー）
pub fn insert_kernel(_key: KernelCacheKey) {
    // 実際の挿入はバックエンド固有の関数を使用
    // この関数は将来の拡張用
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
