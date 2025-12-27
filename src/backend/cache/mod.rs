//! カーネルキャッシュ
//!
//! グラフ単位でコンパイル済みカーネルをキャッシュし、
//! 同一構造のグラフに対してGPUコンパイルをスキップする。

mod store;

pub use store::{
    CacheEntry, CacheStats, KernelCacheKey, get_cache_stats, get_cached_kernel,
    insert_cached_kernel,
};
