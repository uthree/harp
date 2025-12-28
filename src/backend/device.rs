//! PyTorch-like device API
//!
//! Provides a unified device selection API similar to PyTorch's `torch.device`.
//!
//! # Examples
//!
//! ```ignore
//! use harp::backend::{HarpDevice, set_device};
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
use std::sync::Arc;

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
        // 1. Metal (macOS only, feature = "metal")
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if super::metal::MetalDevice::is_available() {
                if let Ok(device) = Self::metal(0) {
                    return Ok(device);
                }
            }
        }

        // 2. OpenCL (feature = "opencl")
        #[cfg(feature = "opencl")]
        {
            if super::opencl::OpenCLDevice::is_available()
                && let Ok(device) = Self::opencl(0)
            {
                return Ok(device);
            }
        }

        // 3. C (feature = "c", always available when enabled)
        #[cfg(feature = "c")]
        {
            return Self::c();
        }

        // No backend available
        #[allow(unreachable_code)]
        Err(DeviceError::NoAvailableBackend)
    }

    /// Metalバックエンドでデバイスを作成
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal(index: usize) -> Result<Self, DeviceError> {
        use super::metal::MetalDevice;

        if !MetalDevice::is_available() {
            return Err(DeviceError::BackendUnavailable {
                backend: DeviceKind::Metal,
                reason: "Metal is not available on this system".to_string(),
            });
        }

        let devices = MetalDevice::list_devices();
        if index >= devices.len() {
            return Err(DeviceError::InvalidDeviceIndex {
                backend: DeviceKind::Metal,
                index,
                available: devices.len(),
            });
        }

        let device = MetalDevice::with_device(index)
            .map_err(|e| DeviceError::InitializationError(e.to_string()))?;

        Ok(Self {
            kind: DeviceKind::Metal,
            index,
            device: Arc::new(device),
        })
    }

    /// Metalバックエンドでデバイスを作成（feature無効時はエラー）
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    pub fn metal(_index: usize) -> Result<Self, DeviceError> {
        Err(DeviceError::BackendUnavailable {
            backend: DeviceKind::Metal,
            reason: "Metal backend is not enabled (requires feature 'metal' on macOS)".to_string(),
        })
    }

    /// OpenCLバックエンドでデバイスを作成
    #[cfg(feature = "opencl")]
    pub fn opencl(index: usize) -> Result<Self, DeviceError> {
        use super::opencl::OpenCLDevice;

        if !OpenCLDevice::is_available() {
            return Err(DeviceError::BackendUnavailable {
                backend: DeviceKind::OpenCL,
                reason: "OpenCL is not available on this system".to_string(),
            });
        }

        let devices = OpenCLDevice::list_devices().unwrap_or_default();
        if index >= devices.len() {
            return Err(DeviceError::InvalidDeviceIndex {
                backend: DeviceKind::OpenCL,
                index,
                available: devices.len(),
            });
        }

        let device = OpenCLDevice::with_device(index)
            .map_err(|e| DeviceError::InitializationError(e.to_string()))?;

        Ok(Self {
            kind: DeviceKind::OpenCL,
            index,
            device: Arc::new(device),
        })
    }

    /// OpenCLバックエンドでデバイスを作成（feature無効時はエラー）
    #[cfg(not(feature = "opencl"))]
    pub fn opencl(_index: usize) -> Result<Self, DeviceError> {
        Err(DeviceError::BackendUnavailable {
            backend: DeviceKind::OpenCL,
            reason: "OpenCL backend is not enabled (requires feature 'opencl')".to_string(),
        })
    }

    /// Cバックエンドでデバイスを作成
    #[cfg(feature = "c")]
    pub fn c() -> Result<Self, DeviceError> {
        use super::c::CDevice;

        let device = CDevice::new();

        Ok(Self {
            kind: DeviceKind::C,
            index: 0,
            device: Arc::new(device),
        })
    }

    /// Cバックエンドでデバイスを作成（feature無効時はエラー）
    #[cfg(not(feature = "c"))]
    pub fn c() -> Result<Self, DeviceError> {
        Err(DeviceError::BackendUnavailable {
            backend: DeviceKind::C,
            reason: "C backend is not enabled (requires feature 'c')".to_string(),
        })
    }

    /// 利用可能なすべてのデバイスを列挙
    ///
    /// # 戻り値
    /// `(DeviceKind, index, device_name)` のベクター
    pub fn list_all() -> Vec<(DeviceKind, usize, String)> {
        let mut devices = Vec::new();

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            use super::metal::MetalDevice;
            if MetalDevice::is_available() {
                for (i, name) in MetalDevice::list_devices().into_iter().enumerate() {
                    devices.push((DeviceKind::Metal, i, name));
                }
            }
        }

        #[cfg(feature = "opencl")]
        {
            use super::opencl::OpenCLDevice;
            if OpenCLDevice::is_available()
                && let Ok(device_names) = OpenCLDevice::list_devices()
            {
                for (i, name) in device_names.into_iter().enumerate() {
                    devices.push((DeviceKind::OpenCL, i, name));
                }
            }
        }

        #[cfg(feature = "c")]
        {
            devices.push((DeviceKind::C, 0, "Generic CPU (C backend)".to_string()));
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
    pub(crate) fn device_arc(&self) -> Arc<dyn Any + Send + Sync> {
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
        match kind {
            DeviceKind::Metal => Self::metal(index),
            DeviceKind::OpenCL => Self::opencl(index),
            DeviceKind::C => {
                if index != 0 {
                    return Err(DeviceError::InvalidDeviceIndex {
                        backend: DeviceKind::C,
                        index,
                        available: 1,
                    });
                }
                Self::c()
            }
            DeviceKind::None => Err(DeviceError::ParseError(
                "Cannot create device with kind 'None'".to_string(),
            )),
        }
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
