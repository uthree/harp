//! カーネルキャッシュ
//!
//! グラフ単位でコンパイル済みカーネルをキャッシュし、
//! 同一構造のグラフに対してGPUコンパイルをスキップする。

mod store;

pub use store::{CacheStats, CachedKernelEntry, KernelCacheKey, get_cached_kernel, insert_kernel};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use store::{MetalCacheEntry, get_metal_kernel, insert_metal_kernel};

#[cfg(feature = "opencl")]
pub use store::{OpenCLCacheEntry, get_opencl_kernel, insert_opencl_kernel};
