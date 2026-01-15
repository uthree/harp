//! Global Device Management
//!
//! PyTorch-like global device management for simplified API.
//! Each thread has its own default device.
//!
//! # Example
//! ```ignore
//! use eclat::backend::global::{set_default_device, get_default_device_kind, with_device, DeviceKind};
//! use eclat::backend::metal::MetalDevice;
//!
//! // Set the default device for this thread
//! let device = MetalDevice::new()?;
//! set_default_device(device);
//!
//! // Check current device kind
//! assert_eq!(get_default_device_kind(), DeviceKind::Metal);
//!
//! // Use a different device for a specific scope
//! with_device(another_device, || {
//!     // This code runs with another_device as default
//! });
//! ```

use super::device::{DeviceError, EclatDevice};
use super::traits::Device;
use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;

/// The kind of device currently set as default
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceKind {
    /// No device is set
    #[default]
    None,
    /// Metal backend (macOS)
    Metal,
    /// OpenCL backend
    OpenCL,
    /// C backend (CPU fallback)
    C,
    /// Rust backend (CPU, experimental)
    Rust,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceKind::None => write!(f, "None"),
            DeviceKind::Metal => write!(f, "Metal"),
            DeviceKind::OpenCL => write!(f, "OpenCL"),
            DeviceKind::C => write!(f, "C"),
            DeviceKind::Rust => write!(f, "Rust"),
        }
    }
}

/// Global device state for each thread
struct GlobalDeviceState {
    kind: DeviceKind,
    device: Option<Arc<dyn Any + Send + Sync>>,
}

impl Default for GlobalDeviceState {
    fn default() -> Self {
        Self {
            kind: DeviceKind::None,
            device: None,
        }
    }
}

thread_local! {
    static DEFAULT_DEVICE: RefCell<GlobalDeviceState> = RefCell::new(GlobalDeviceState::default());
}

/// Get the kind of the default device for this thread
pub fn get_default_device_kind() -> DeviceKind {
    DEFAULT_DEVICE.with(|state| state.borrow().kind)
}

/// Check if a default device is set for this thread
pub fn has_default_device() -> bool {
    get_default_device_kind() != DeviceKind::None
}

/// Set the default device for this thread
///
/// # Arguments
/// * `device` - The device to set as default
/// * `kind` - The kind of device (Metal or OpenCL)
///
/// # Example
/// ```ignore
/// use eclat::backend::global::{set_default_device, DeviceKind};
/// use eclat::backend::metal::MetalDevice;
///
/// let device = MetalDevice::new()?;
/// set_default_device(device, DeviceKind::Metal);
/// ```
pub fn set_default_device<D: Device + Send + Sync + 'static>(device: D, kind: DeviceKind) {
    DEFAULT_DEVICE.with(|state| {
        let mut state = state.borrow_mut();
        state.kind = kind;
        state.device = Some(Arc::new(device));
    });
}

/// Set the default device using EclatDevice
///
/// This is the recommended way to set the default device.
///
/// # Example
/// ```ignore
/// use eclat::backend::{EclatDevice, set_device};
///
/// let device = EclatDevice::auto()?;
/// set_device(device);
/// ```
pub fn set_device(device: EclatDevice) {
    DEFAULT_DEVICE.with(|state| {
        let mut state = state.borrow_mut();
        state.kind = device.kind();
        state.device = Some(device.device_arc());
    });
}

/// Set the default device using a device string
///
/// # Example
/// ```ignore
/// use eclat::backend::set_device_str;
///
/// set_device_str("opencl:0")?;
/// set_device_str("metal")?;
/// set_device_str("c")?;
/// ```
pub fn set_device_str(device_str: &str) -> Result<(), DeviceError> {
    let device = EclatDevice::new(device_str)?;
    set_device(device);
    Ok(())
}

/// Clear the default device for this thread
pub fn clear_default_device() {
    DEFAULT_DEVICE.with(|state| {
        let mut state = state.borrow_mut();
        state.kind = DeviceKind::None;
        state.device = None;
    });
}

/// Get the default device for this thread, if it matches the expected type
///
/// Returns None if no device is set or if the device is not of the expected type.
pub fn get_default_device<D: Device + Send + Sync + 'static>() -> Option<Arc<D>> {
    DEFAULT_DEVICE.with(|state| {
        let state = state.borrow();
        state
            .device
            .as_ref()
            .and_then(|d| d.clone().downcast::<D>().ok())
    })
}

/// デフォルトデバイス上にバッファを割り当てる
///
/// # Arguments
/// * `shape` - バッファの形状
/// * `dtype` - データ型
///
/// # Returns
/// 割り当てられたバッファ、またはエラー
///
/// # Errors
/// - デバイスが設定されていない場合
/// - バックエンドがランタイム実行をサポートしない場合
/// - バッファ割り当てに失敗した場合
pub fn allocate_buffer_on_default_device(
    shape: Vec<usize>,
    dtype: crate::ast::DType,
) -> Result<Box<dyn crate::backend::Buffer>, super::device::DeviceError> {
    use super::device::{DeviceError, allocate_buffer_on_device};

    let (kind, device) = DEFAULT_DEVICE.with(|state| {
        let state = state.borrow();
        (state.kind, state.device.clone())
    });

    if kind == DeviceKind::None {
        return Err(DeviceError::NoAvailableBackend);
    }

    let device = device.ok_or(DeviceError::NoAvailableBackend)?;
    allocate_buffer_on_device(kind, device.as_ref(), shape, dtype)
}

/// デフォルトデバイス上でASTをコンパイルしてカーネルを返す
///
/// # Arguments
/// * `program` - コンパイルするASTプログラム
/// * `signature` - カーネルの入出力シグネチャ
///
/// # Returns
/// コンパイルされたカーネル、またはエラー
pub fn compile_ast_on_default_device(
    program: crate::ast::AstNode,
    signature: crate::backend::KernelSignature,
) -> Result<Box<dyn crate::backend::Kernel>, super::device::DeviceError> {
    use super::device::{DeviceError, compile_ast_on_device};

    let (kind, device) = DEFAULT_DEVICE.with(|state| {
        let state = state.borrow();
        (state.kind, state.device.clone())
    });

    if kind == DeviceKind::None {
        return Err(DeviceError::NoAvailableBackend);
    }

    let device = device.ok_or(DeviceError::NoAvailableBackend)?;
    compile_ast_on_device(kind, device.as_ref(), program, signature)
}

/// キャッシュ対応のASTコンパイル関数
///
/// 同一構造のASTに対してはキャッシュされたカーネルを返し、
/// GPUコンパイルをスキップする。
///
/// キャッシュ階層:
/// 1. メモリキャッシュ（同一プロセス内）
/// 2. ディスクキャッシュ（セッション間、OpenCLのみ）
///
/// # Arguments
/// * `program` - コンパイルするASTプログラム
/// * `signature` - カーネルの入出力シグネチャ
///
/// # Returns
/// キャッシュエントリ（カーネル、シグネチャ、ディスパッチ設定を含む）
pub fn compile_ast_with_cache(
    program: crate::ast::AstNode,
    signature: crate::backend::KernelSignature,
) -> Result<crate::backend::cache::CacheEntry, super::device::DeviceError> {
    use super::cache::disk::{
        DiskCacheMetadata, compute_cache_hash, eclat_version, load_binary, save_binary,
    };
    use super::cache::{generate_cache_key, get_cached_kernel, insert_cached_kernel};
    use super::device::{DeviceError, compile_from_binary_on_device};

    let (kind, device) = DEFAULT_DEVICE.with(|state| {
        let state = state.borrow();
        (state.kind, state.device.clone())
    });

    if kind == DeviceKind::None {
        return Err(DeviceError::NoAvailableBackend);
    }

    let device = device.ok_or(DeviceError::NoAvailableBackend)?;

    // dyn traitはファットポインタなので、データポインタ部分のみを取得
    let device_id = Arc::as_ptr(&device) as *const () as usize;

    // キャッシュキー生成
    let cache_key = generate_cache_key(&program, kind, device_id);

    // 1. メモリキャッシュをチェック
    if let Some(entry) = get_cached_kernel(&cache_key) {
        log::debug!("Memory cache hit");
        return Ok(entry);
    }

    // 2. ディスクキャッシュをチェック（OpenCLのみ）
    let disk_hash = compute_cache_hash(&cache_key);
    let device_kind_str = kind.to_string();

    if let Some((binary, metadata)) = load_binary(&disk_hash, &device_kind_str) {
        // バージョンチェック
        if metadata.eclat_version == eclat_version() {
            log::debug!("Disk cache hit: {}", disk_hash);

            // バイナリからカーネルを復元
            let config = crate::backend::KernelConfig {
                entry_point: metadata.entry_point,
                global_work_size: metadata.grid_size,
                local_work_size: metadata.local_size,
                shape_vars: std::collections::HashMap::new(),
            };

            if let Ok(kernel) =
                compile_from_binary_on_device(kind, device.as_ref(), &binary, config)
            {
                let dispatch_config = crate::backend::pipeline::DispatchSizeConfig::from_const(
                    metadata.grid_size,
                    metadata.local_size.unwrap_or([1, 1, 1]),
                );
                let entry = crate::backend::cache::CacheEntry::new(
                    kernel,
                    signature.clone(),
                    dispatch_config,
                );

                // メモリキャッシュにも挿入
                insert_cached_kernel(cache_key, entry.clone());

                return Ok(entry);
            } else {
                log::debug!("Failed to restore from disk cache, recompiling...");
            }
        } else {
            log::debug!(
                "Disk cache version mismatch: {} != {}",
                metadata.eclat_version,
                eclat_version()
            );
        }
    }

    log::debug!("Cache miss, compiling...");

    // 3. コンパイル
    let kernel = compile_ast_on_default_device(program, signature.clone())?;

    // CacheEntryを構築（デフォルトのディスパッチ設定を使用）
    let dispatch_config =
        crate::backend::pipeline::DispatchSizeConfig::from_const([1, 1, 1], [1, 1, 1]);
    let entry =
        crate::backend::cache::CacheEntry::new(kernel, signature.clone(), dispatch_config.clone());

    // 4. ディスクキャッシュに保存（バイナリサポートがある場合）
    if entry.kernel.supports_binary_cache()
        && let Some(binary) = entry.kernel.get_binary() {
            let metadata = DiskCacheMetadata {
                entry_point: entry.kernel.config().entry_point.clone(),
                grid_size: entry.kernel.config().global_work_size,
                local_size: entry.kernel.config().local_work_size,
                device_name: kind.to_string(),
                eclat_version: eclat_version().to_string(),
            };

            if let Err(e) = save_binary(&disk_hash, &device_kind_str, &binary, &metadata) {
                log::debug!("Failed to save disk cache: {}", e);
            } else {
                log::debug!(
                    "Saved to disk cache: {} ({} bytes)",
                    disk_hash,
                    binary.len()
                );
            }
        }

    // 5. メモリキャッシュに挿入
    insert_cached_kernel(cache_key, entry.clone());

    Ok(entry)
}

/// Run a closure with a temporary default device
///
/// The original default device is restored after the closure completes.
///
/// # Example
/// ```ignore
/// use eclat::backend::global::{with_device, DeviceKind};
///
/// with_device(my_device, DeviceKind::Metal, || {
///     // Code here uses my_device as the default
///     let tensor = Tensor::zeros([10, 10]);
///     tensor.forward()?;
///     Ok(())
/// })?;
/// // Original device is restored here
/// ```
pub fn with_device<D, F, R>(device: D, kind: DeviceKind, f: F) -> R
where
    D: Device + Send + Sync + 'static,
    F: FnOnce() -> R,
{
    // Save current state
    let (old_kind, old_device) = DEFAULT_DEVICE.with(|state| {
        let state = state.borrow();
        (state.kind, state.device.clone())
    });

    // Set new device
    set_default_device(device, kind);

    // Run closure
    let result = f();

    // Restore old state
    DEFAULT_DEVICE.with(|state| {
        let mut state = state.borrow_mut();
        state.kind = old_kind;
        state.device = old_device;
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_device_kind_is_none() {
        // Reset state for this test
        clear_default_device();
        assert_eq!(get_default_device_kind(), DeviceKind::None);
        assert!(!has_default_device());
    }

    #[test]
    fn test_device_kind_display() {
        assert_eq!(format!("{}", DeviceKind::None), "None");
        assert_eq!(format!("{}", DeviceKind::Metal), "Metal");
        assert_eq!(format!("{}", DeviceKind::OpenCL), "OpenCL");
        assert_eq!(format!("{}", DeviceKind::C), "C");
        assert_eq!(format!("{}", DeviceKind::Rust), "Rust");
    }
}
