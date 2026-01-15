//! CUDA device implementation
//!
//! Manages CUDA context and provides device information.

use eclat::backend::traits::Device;
use std::process::Command;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaContext, CudaStream, DriverError};
#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

/// CUDA device for GPU execution
#[cfg(feature = "cuda-runtime")]
pub struct CudaDevice {
    /// CUDA context
    context: Arc<CudaContext>,
    /// Default stream for this device
    stream: CudaStream,
    /// Device index
    device_index: usize,
}

/// Stub CUDA device (when cuda-runtime feature is disabled)
#[cfg(not(feature = "cuda-runtime"))]
pub struct CudaDevice {
    /// Device index
    device_index: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaDevice {
    /// Create a new CUDA device
    pub fn new(device_index: usize) -> Result<Self, CudaDeviceError> {
        let context = CudaContext::new(device_index).map_err(|e| {
            CudaDeviceError::InitializationError(format!(
                "Failed to create CUDA context for device {}: {}",
                device_index, e
            ))
        })?;

        let stream = context.default_stream();

        Ok(Self {
            context,
            stream,
            device_index,
        })
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get the default stream
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// List available CUDA devices
    pub fn list_devices() -> Vec<String> {
        let mut devices = Vec::new();
        for i in 0..16 {
            if CudaContext::new(i).is_ok() {
                devices.push(format!("CUDA Device {}", i));
            } else {
                break;
            }
        }
        devices
    }
}

#[cfg(not(feature = "cuda-runtime"))]
impl CudaDevice {
    /// Create a new CUDA device (stub - always fails without cuda-runtime feature)
    pub fn new(_device_index: usize) -> Result<Self, CudaDeviceError> {
        Err(CudaDeviceError::InitializationError(
            "CUDA runtime not available. Rebuild with --features cuda-runtime".to_string()
        ))
    }

    /// List available CUDA devices (empty without cuda-runtime feature)
    pub fn list_devices() -> Vec<String> {
        Vec::new()
    }
}

impl CudaDevice {
    /// Check if nvcc compiler is available
    pub fn is_nvcc_available() -> bool {
        Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Get the device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }
}

#[cfg(feature = "cuda-runtime")]
impl Clone for CudaDevice {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            stream: self.context.default_stream(),
            device_index: self.device_index,
        }
    }
}

#[cfg(not(feature = "cuda-runtime"))]
impl Clone for CudaDevice {
    fn clone(&self) -> Self {
        Self {
            device_index: self.device_index,
        }
    }
}

impl Device for CudaDevice {
    #[cfg(feature = "cuda-runtime")]
    fn is_available() -> bool {
        if !Self::is_nvcc_available() {
            return false;
        }
        CudaContext::new(0).is_ok()
    }

    #[cfg(not(feature = "cuda-runtime"))]
    fn is_available() -> bool {
        false
    }
}

/// Error types for CUDA device operations
#[derive(Debug)]
pub enum CudaDeviceError {
    /// Device initialization failed
    InitializationError(String),
    /// Driver error
    #[cfg(feature = "cuda-runtime")]
    DriverError(DriverError),
}

impl std::fmt::Display for CudaDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaDeviceError::InitializationError(msg) => write!(f, "CUDA initialization error: {}", msg),
            #[cfg(feature = "cuda-runtime")]
            CudaDeviceError::DriverError(e) => write!(f, "CUDA driver error: {}", e),
        }
    }
}

impl std::error::Error for CudaDeviceError {}

#[cfg(feature = "cuda-runtime")]
impl From<DriverError> for CudaDeviceError {
    fn from(e: DriverError) -> Self {
        CudaDeviceError::DriverError(e)
    }
}
