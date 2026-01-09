//! PyTorch-like device API
//!
//! Provides a unified device selection API similar to PyTorch's `torch.device`.
//!
//! # Examples
//!
//! ```ignore
//! use eclat::backend::{HarpDevice, set_device};
//!
//! // Auto-select best available backend (Metal > OpenCL > C)
//! let device = HarpDevice::auto()?;
//! set_device(device);
//!
//! // Explicit backend selection
//! let device = HarpDevice::new("metal")?;
//! let device = HarpDevice::new("opencl:0")?;  // with device index
//! let device = HarpDevice::new("c")?;
//!
//! // List all available devices
//! for (kind, index, name) in HarpDevice::list_all() {
//!     println!("{:?}:{} - {}", kind, index, name);
//! }
//! ```

use super::global::DeviceKind;
use super::traits::Device;
use std::any::Any;
use std::sync::{Arc, OnceLock, RwLock};

/// デバイス関連のエラー
#[derive(Debug)]
pub enum DeviceError {
    /// 指定されたバックエンドが利用できない
    BackendUnavailable { backend: DeviceKind, reason: String },
    /// デバイスインデックスが無効
    InvalidDeviceIndex {
        backend: DeviceKind,
        index: usize,
        available: usize,
    },
    /// デバイス文字列のパースエラー
    ParseError(String),
    /// 利用可能なバックエンドがない
    NoAvailableBackend,
    /// バックエンド初期化エラー
    InitializationError(String),
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceError::BackendUnavailable { backend, reason } => {
                write!(f, "Backend {:?} is unavailable: {}", backend, reason)
            }
            DeviceError::InvalidDeviceIndex {
                backend,
                index,
                available,
            } => {
                write!(
                    f,
                    "Invalid device index {} for {:?} backend (available: {})",
                    index, backend, available
                )
            }
            DeviceError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            DeviceError::NoAvailableBackend => write!(f, "No available backend"),
            DeviceError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
        }
    }
}

impl std::error::Error for DeviceError {}

// ============================================================================
// Backend Registry
// ============================================================================

/// バックエンド登録用のトレイト
///
/// 各バックエンドクレート（eclat-backend-c, eclat-backend-metal, eclat-backend-opencl）は
/// このトレイトを実装し、`register_backend` 関数で登録する。
pub trait BackendRegistry: Send + Sync {
    /// バックエンドの種類
    fn kind(&self) -> DeviceKind;

    /// バックエンドの名前
    fn name(&self) -> &str;

    /// このバックエンドが利用可能かどうか
    fn is_available(&self) -> bool;

    /// デバイスを作成
    fn create_device(&self, index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError>;

    /// 利用可能なデバイスの一覧を取得
    fn list_devices(&self) -> Vec<String>;
}

/// 登録されたバックエンドのグローバルレジストリ
static BACKENDS: OnceLock<RwLock<Vec<Box<dyn BackendRegistry>>>> = OnceLock::new();

/// バックエンドを登録する
///
/// バックエンドクレートの初期化時に呼び出される。
/// 登録されたバックエンドは `HarpDevice::auto()` や `HarpDevice::new()` で使用される。
pub fn register_backend(backend: Box<dyn BackendRegistry>) {
    let backends = BACKENDS.get_or_init(|| RwLock::new(Vec::new()));
    let mut backends = backends.write().unwrap();

    // 同じ種類のバックエンドが既に登録されていないか確認
    let kind = backend.kind();
    if !backends.iter().any(|b| b.kind() == kind) {
        log::debug!("Registering backend: {:?} ({})", kind, backend.name());
        backends.push(backend);
    }
}

/// 登録されたバックエンドを取得
fn get_backends() -> std::sync::RwLockReadGuard<'static, Vec<Box<dyn BackendRegistry>>> {
    let backends = BACKENDS.get_or_init(|| RwLock::new(Vec::new()));
    backends.read().unwrap()
}

// ============================================================================
// HarpDevice
// ============================================================================

/// PyTorch風のデバイス指定
///
/// 文字列または明示的なコンストラクタでデバイスを指定できる。
/// 自動選択機能も提供。
///
/// # 使用例
///
/// ```ignore
/// // 文字列から作成
/// let device = HarpDevice::new("metal")?;
/// let device = HarpDevice::new("opencl:0")?;
/// let device = HarpDevice::new("c")?;
///
/// // 自動選択（優先順位: Metal > OpenCL > C）
/// let device = HarpDevice::auto()?;
///
/// // デフォルトとして設定
/// device.set_as_default();
/// ```
#[derive(Clone)]
pub struct HarpDevice {
    kind: DeviceKind,
    index: usize,
    device: Arc<dyn Any + Send + Sync>,
}

impl std::fmt::Debug for HarpDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarpDevice")
            .field("kind", &self.kind)
            .field("index", &self.index)
            .finish()
    }
}

impl HarpDevice {
    /// デバイスを直接作成（バックエンドクレートから呼び出される）
    pub fn from_device(kind: DeviceKind, index: usize, device: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            kind,
            index,
            device,
        }
    }

    /// 文字列からデバイスを作成
    ///
    /// # フォーマット
    /// - `"metal"` - Metal backend, device 0
    /// - `"opencl"` - OpenCL backend, device 0
    /// - `"opencl:1"` - OpenCL backend, device 1
    /// - `"c"` - C backend
    ///
    /// # エラー
    /// - 不明なバックエンド名
    /// - 無効なデバイスインデックス
    /// - バックエンドが利用できない
    pub fn new(device_str: &str) -> Result<Self, DeviceError> {
        let (backend_str, index) = Self::parse_device_str(device_str)?;
        let kind = Self::parse_backend(backend_str)?;
        Self::create_for_kind(kind, index)
    }

    /// 優先順位に基づいて最適なバックエンドを自動選択
    ///
    /// # 優先順位
    /// 1. Metal (macOSでネイティブGPU)
    /// 2. OpenCL (クロスプラットフォームGPU)
    /// 3. C (フォールバック)
    ///
    /// # エラー
    /// - 利用可能なバックエンドがない場合
    pub fn auto() -> Result<Self, DeviceError> {
        let backends = get_backends();

        // 優先順位: Metal > OpenCL > C
        for kind in [DeviceKind::Metal, DeviceKind::OpenCL, DeviceKind::C] {
            if let Some(backend) = backends
                .iter()
                .find(|b| b.kind() == kind && b.is_available())
            {
                let device = backend.create_device(0)?;
                return Ok(Self {
                    kind,
                    index: 0,
                    device,
                });
            }
        }

        Err(DeviceError::NoAvailableBackend)
    }

    /// Metalバックエンドでデバイスを作成
    pub fn metal(index: usize) -> Result<Self, DeviceError> {
        Self::create_for_kind(DeviceKind::Metal, index)
    }

    /// OpenCLバックエンドでデバイスを作成
    pub fn opencl(index: usize) -> Result<Self, DeviceError> {
        Self::create_for_kind(DeviceKind::OpenCL, index)
    }

    /// Cバックエンドでデバイスを作成
    pub fn c() -> Result<Self, DeviceError> {
        Self::create_for_kind(DeviceKind::C, 0)
    }

    /// 利用可能なすべてのデバイスを列挙
    ///
    /// # 戻り値
    /// `(DeviceKind, index, device_name)` のベクター
    pub fn list_all() -> Vec<(DeviceKind, usize, String)> {
        let backends = get_backends();
        let mut devices = Vec::new();

        for backend in backends.iter() {
            if backend.is_available() {
                for (i, name) in backend.list_devices().into_iter().enumerate() {
                    devices.push((backend.kind(), i, name));
                }
            }
        }

        devices
    }

    /// デバイスの種類を取得
    pub fn kind(&self) -> DeviceKind {
        self.kind
    }

    /// デバイスインデックスを取得
    pub fn index(&self) -> usize {
        self.index
    }

    /// このデバイスをスレッドのデフォルトとして設定
    pub fn set_as_default(self) {
        super::global::set_device(self);
    }

    /// 内部デバイスをダウンキャストして取得
    pub fn get_device<D: Device + Send + Sync + 'static>(&self) -> Option<Arc<D>> {
        self.device.clone().downcast::<D>().ok()
    }

    /// 内部デバイスのArcを取得（型消去）
    pub fn device_arc(&self) -> Arc<dyn Any + Send + Sync> {
        self.device.clone()
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// デバイス文字列をパース
    fn parse_device_str(s: &str) -> Result<(&str, usize), DeviceError> {
        if let Some((backend, idx_str)) = s.split_once(':') {
            let index = idx_str.parse::<usize>().map_err(|_| {
                DeviceError::ParseError(format!("Invalid device index: {}", idx_str))
            })?;
            Ok((backend, index))
        } else {
            Ok((s, 0))
        }
    }

    /// バックエンド名をパース
    fn parse_backend(s: &str) -> Result<DeviceKind, DeviceError> {
        match s.to_lowercase().as_str() {
            "metal" => Ok(DeviceKind::Metal),
            "opencl" => Ok(DeviceKind::OpenCL),
            "c" => Ok(DeviceKind::C),
            _ => Err(DeviceError::ParseError(format!(
                "Unknown backend: '{}'. Valid options: metal, opencl, c",
                s
            ))),
        }
    }

    /// DeviceKindに対応するデバイスを作成
    fn create_for_kind(kind: DeviceKind, index: usize) -> Result<Self, DeviceError> {
        let backends = get_backends();

        // 対応するバックエンドを検索
        let backend = backends.iter().find(|b| b.kind() == kind).ok_or_else(|| {
            DeviceError::BackendUnavailable {
                backend: kind,
                reason: format!(
                    "Backend not registered. Make sure eclat-backend-{} is included as a dependency.",
                    kind.to_string().to_lowercase()
                ),
            }
        })?;

        // 利用可能かチェック
        if !backend.is_available() {
            return Err(DeviceError::BackendUnavailable {
                backend: kind,
                reason: format!("{} is not available on this system", backend.name()),
            });
        }

        // デバイスを作成
        let device = backend.create_device(index)?;
        Ok(Self {
            kind,
            index,
            device,
        })
    }
}

impl std::str::FromStr for HarpDevice {
    type Err = DeviceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device_str() {
        assert_eq!(HarpDevice::parse_device_str("metal").unwrap(), ("metal", 0));
        assert_eq!(
            HarpDevice::parse_device_str("opencl:0").unwrap(),
            ("opencl", 0)
        );
        assert_eq!(
            HarpDevice::parse_device_str("opencl:1").unwrap(),
            ("opencl", 1)
        );
        assert_eq!(HarpDevice::parse_device_str("c").unwrap(), ("c", 0));
    }

    #[test]
    fn test_parse_backend() {
        assert_eq!(
            HarpDevice::parse_backend("metal").unwrap(),
            DeviceKind::Metal
        );
        assert_eq!(
            HarpDevice::parse_backend("Metal").unwrap(),
            DeviceKind::Metal
        );
        assert_eq!(
            HarpDevice::parse_backend("METAL").unwrap(),
            DeviceKind::Metal
        );
        assert_eq!(
            HarpDevice::parse_backend("opencl").unwrap(),
            DeviceKind::OpenCL
        );
        assert_eq!(HarpDevice::parse_backend("c").unwrap(), DeviceKind::C);
        assert!(HarpDevice::parse_backend("unknown").is_err());
    }

    #[test]
    fn test_device_error_display() {
        let err = DeviceError::BackendUnavailable {
            backend: DeviceKind::Metal,
            reason: "test reason".to_string(),
        };
        assert!(err.to_string().contains("Metal"));
        assert!(err.to_string().contains("test reason"));
    }
}
