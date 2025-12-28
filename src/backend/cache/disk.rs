//! ディスクキャッシュ
//!
//! コンパイル済みカーネルバイナリをディスクに保存し、
//! セッション間で再利用できるようにする。

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{self, ErrorKind};
use std::path::PathBuf;

use super::KernelCacheKey;

/// ディスクキャッシュのメタデータ
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskCacheMetadata {
    /// カーネルエントリポイント名
    pub entry_point: String,
    /// グローバルワークサイズ
    pub grid_size: [usize; 3],
    /// ローカルワークサイズ
    pub local_size: Option<[usize; 3]>,
    /// デバイス名（バイナリ互換性チェック用）
    pub device_name: String,
    /// Harpバージョン
    pub harp_version: String,
}

/// キャッシュディレクトリを取得
///
/// 環境変数で制御可能:
/// - `HARP_NO_DISK_CACHE=1`: ディスクキャッシュ無効化
/// - `HARP_CACHE_DIR`: カスタムキャッシュディレクトリ
pub fn get_cache_dir() -> Option<PathBuf> {
    // 環境変数でディスクキャッシュを無効化
    if std::env::var("HARP_NO_DISK_CACHE").is_ok() {
        return None;
    }

    // カスタムキャッシュディレクトリ
    if let Some(dir) = std::env::var_os("HARP_CACHE_DIR") {
        return Some(PathBuf::from(dir).join("kernels"));
    }

    // システム標準のキャッシュディレクトリ
    directories::ProjectDirs::from("", "", "harp").map(|dirs| dirs.cache_dir().join("kernels"))
}

/// キャッシュキーからハッシュを計算
///
/// 注意: メモリキャッシュとは異なり、device_idはプロセスごとに変わるため使用しない。
/// 代わりにgraph_reprとdevice_kindのみを使用する。
/// デバイスの互換性はロード時にdevice_nameで検証する。
pub fn compute_cache_hash(key: &KernelCacheKey) -> String {
    let mut hasher = DefaultHasher::new();
    // graph_reprのみをハッシュに使用（device_idは除外）
    key.graph_repr().hash(&mut hasher);
    key.device_kind().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// ディスクからバイナリとメタデータを読み込み
pub fn load_binary(hash: &str, device_kind: &str) -> Option<(Vec<u8>, DiskCacheMetadata)> {
    let cache_dir = get_cache_dir()?;
    let bin_path = cache_dir.join(device_kind).join(format!("{}.bin", hash));
    let meta_path = cache_dir.join(device_kind).join(format!("{}.json", hash));

    // バイナリ読み込み
    let binary = std::fs::read(&bin_path).ok()?;

    // メタデータ読み込み
    let meta_str = std::fs::read_to_string(&meta_path).ok()?;
    let metadata: DiskCacheMetadata = serde_json::from_str(&meta_str).ok()?;

    log::debug!(
        "Disk cache hit: {} (entry_point={})",
        hash,
        metadata.entry_point
    );

    Some((binary, metadata))
}

/// バイナリとメタデータをディスクに保存
pub fn save_binary(
    hash: &str,
    device_kind: &str,
    binary: &[u8],
    metadata: &DiskCacheMetadata,
) -> io::Result<()> {
    let cache_dir = get_cache_dir()
        .ok_or_else(|| io::Error::new(ErrorKind::NotFound, "disk cache disabled"))?;

    let dir = cache_dir.join(device_kind);
    std::fs::create_dir_all(&dir)?;

    // バイナリ保存
    let bin_path = dir.join(format!("{}.bin", hash));
    std::fs::write(&bin_path, binary)?;

    // メタデータ保存
    let meta_path = dir.join(format!("{}.json", hash));
    let meta_json = serde_json::to_string_pretty(metadata)
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
    std::fs::write(&meta_path, meta_json)?;

    log::debug!(
        "Disk cache saved: {} ({} bytes, entry_point={})",
        hash,
        binary.len(),
        metadata.entry_point
    );

    Ok(())
}

/// 現在のHarpバージョンを取得
pub fn harp_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::global::DeviceKind;

    #[test]
    fn test_compute_cache_hash() {
        let key1 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::OpenCL, 0x1000);
        let key2 = KernelCacheKey::new("Add($0, $1)".to_string(), DeviceKind::OpenCL, 0x1000);
        let key3 = KernelCacheKey::new("Mul($0, $1)".to_string(), DeviceKind::OpenCL, 0x1000);

        // 同じキーは同じハッシュ
        assert_eq!(compute_cache_hash(&key1), compute_cache_hash(&key2));

        // 異なるキーは異なるハッシュ
        assert_ne!(compute_cache_hash(&key1), compute_cache_hash(&key3));
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = DiskCacheMetadata {
            entry_point: "kernel_main".to_string(),
            grid_size: [256, 1, 1],
            local_size: Some([64, 1, 1]),
            device_name: "Apple M1".to_string(),
            harp_version: "0.1.0".to_string(),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: DiskCacheMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.entry_point, metadata.entry_point);
        assert_eq!(deserialized.grid_size, metadata.grid_size);
        assert_eq!(deserialized.local_size, metadata.local_size);
        assert_eq!(deserialized.device_name, metadata.device_name);
    }

    #[test]
    fn test_get_cache_dir_env_disabled() {
        // 環境変数でディスクキャッシュを無効化
        // SAFETY: テスト環境でのみ使用。並行テストでは競合の可能性あり。
        unsafe {
            std::env::set_var("HARP_NO_DISK_CACHE", "1");
        }
        assert!(get_cache_dir().is_none());
        unsafe {
            std::env::remove_var("HARP_NO_DISK_CACHE");
        }
    }
}
