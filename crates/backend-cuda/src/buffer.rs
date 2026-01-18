//! CUDA buffer implementation
//!
//! Provides GPU memory management for CUDA backend.

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::device::CudaDevice;
use eclat::ast::DType;
use eclat::backend::traits::Buffer;
use std::any::Any;

/// CUDA buffer for GPU memory
#[cfg(feature = "cuda-runtime")]
pub struct CudaBuffer {
    /// Device memory slice (stored as bytes)
    data: CudaSlice<u8>,
    /// Shape of the buffer
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Total size in bytes
    size_bytes: usize,
}

/// Stub CUDA buffer (when cuda-runtime feature is disabled)
#[cfg(not(feature = "cuda-runtime"))]
pub struct CudaBuffer {
    /// Shape of the buffer
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Total size in bytes
    size_bytes: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaBuffer {
    /// Allocate a new CUDA buffer
    pub fn allocate(
        device: &CudaDevice,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, CudaBufferError> {
        let num_elements: usize = shape.iter().product();
        let element_size = dtype.size_bytes();
        let size_bytes = num_elements * element_size;

        if size_bytes == 0 {
            return Err(CudaBufferError::AllocationError(
                "Cannot allocate zero-sized buffer".to_string(),
            ));
        }

        let stream = device.stream();
        let data = stream.alloc_zeros::<u8>(size_bytes).map_err(|e| {
            CudaBufferError::AllocationError(format!("CUDA allocation failed: {}", e))
        })?;

        Ok(Self {
            data,
            shape,
            dtype,
            size_bytes,
        })
    }

    /// Get the raw device pointer as u64 (for kernel arguments)
    pub fn as_device_ptr_u64(&self) -> u64 {
        *self.data.device_ptr() as u64
    }

    /// Copy data from host to device
    pub fn copy_from_host(
        &mut self,
        device: &CudaDevice,
        data: &[u8],
    ) -> Result<(), CudaBufferError> {
        if data.len() != self.size_bytes {
            return Err(CudaBufferError::SizeMismatch {
                expected: self.size_bytes,
                actual: data.len(),
            });
        }

        let stream = device.stream();
        stream.memcpy_htod(&data, &mut self.data).map_err(|e| {
            CudaBufferError::CopyError(format!("Host to device copy failed: {}", e))
        })?;

        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(
        &self,
        device: &CudaDevice,
        data: &mut [u8],
    ) -> Result<(), CudaBufferError> {
        if data.len() != self.size_bytes {
            return Err(CudaBufferError::SizeMismatch {
                expected: self.size_bytes,
                actual: data.len(),
            });
        }

        let stream = device.stream();
        stream.memcpy_dtoh(&self.data, data).map_err(|e| {
            CudaBufferError::CopyError(format!("Device to host copy failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(not(feature = "cuda-runtime"))]
impl CudaBuffer {
    /// Allocate a new CUDA buffer (stub - always fails)
    pub fn allocate(
        _device: &CudaDevice,
        _shape: Vec<usize>,
        _dtype: DType,
    ) -> Result<Self, CudaBufferError> {
        Err(CudaBufferError::AllocationError(
            "CUDA runtime not available. Rebuild with --features cuda-runtime".to_string(),
        ))
    }

    /// Get the raw device pointer (stub)
    pub fn as_device_ptr_u64(&self) -> u64 {
        0
    }
}

impl Buffer for CudaBuffer {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn byte_len(&self) -> usize {
        self.size_bytes
    }

    fn read_to_host(&self) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        Err("CudaBuffer::read_to_host requires device context".into())
    }

    fn write_from_host(
        &mut self,
        _data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Err("CudaBuffer::write_from_host requires device context".into())
    }

    fn clone_buffer(&self) -> Box<dyn Buffer> {
        Box::new(CudaBuffer {
            #[cfg(feature = "cuda-runtime")]
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            size_bytes: self.size_bytes,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Error types for CUDA buffer operations
#[derive(Debug)]
pub enum CudaBufferError {
    /// Memory allocation failed
    AllocationError(String),
    /// Size mismatch during copy
    SizeMismatch { expected: usize, actual: usize },
    /// Copy operation failed
    CopyError(String),
}

impl std::fmt::Display for CudaBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaBufferError::AllocationError(msg) => write!(f, "CUDA allocation error: {}", msg),
            CudaBufferError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Size mismatch: expected {} bytes, got {}",
                    expected, actual
                )
            }
            CudaBufferError::CopyError(msg) => write!(f, "CUDA copy error: {}", msg),
        }
    }
}

impl std::error::Error for CudaBufferError {}
