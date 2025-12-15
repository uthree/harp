//! Metal native buffer

use super::context::{MetalNativeContext, MetalNativeError};
use crate::ast::DType;
use crate::backend::traits::NativeBuffer;
use metal::{Buffer as MtlBuffer, MTLResourceOptions};
use std::sync::Arc;

/// Metal native buffer
///
/// Wraps a Metal buffer with shape and type information.
#[derive(Clone)]
pub struct MetalNativeBuffer {
    buffer: Arc<MtlBuffer>,
    shape: Vec<usize>,
    dtype: DType,
    byte_len: usize,
}

impl NativeBuffer for MetalNativeBuffer {
    type Context = MetalNativeContext;
    type Error = MetalNativeError;

    fn allocate(
        context: &Self::Context,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, Self::Error> {
        let element_count: usize = shape.iter().product();
        let byte_len = element_count * dtype.size_in_bytes();

        // Create Metal buffer with shared storage mode for CPU/GPU access
        let buffer = context
            .device()
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            dtype,
            byte_len,
        })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn write_from_host(&mut self, data: &[u8]) -> Result<(), Self::Error> {
        if data.len() != self.byte_len {
            return Err(Self::buffer_size_mismatch_error(data.len(), self.byte_len));
        }

        // Copy data to buffer
        let buffer_ptr = self.buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer_ptr, data.len());
        }

        Ok(())
    }

    fn read_to_host(&self) -> Result<Vec<u8>, Self::Error> {
        let mut result = vec![0u8; self.byte_len];
        let buffer_ptr = self.buffer.contents() as *const u8;
        unsafe {
            std::ptr::copy_nonoverlapping(buffer_ptr, result.as_mut_ptr(), self.byte_len);
        }
        Ok(result)
    }

    fn buffer_size_mismatch_error(expected: usize, actual: usize) -> Self::Error {
        MetalNativeError::from(format!(
            "Buffer size mismatch: expected {} bytes, got {} bytes",
            expected, actual
        ))
    }

    fn buffer_alignment_error(buffer_size: usize, type_size: usize) -> Self::Error {
        MetalNativeError::from(format!(
            "Buffer size {} is not aligned to type size {}",
            buffer_size, type_size
        ))
    }
}

impl MetalNativeBuffer {
    /// Get the underlying Metal buffer
    pub fn mtl_buffer(&self) -> &MtlBuffer {
        &self.buffer
    }

    /// Create a buffer initialized with data from host
    pub fn from_host(
        context: &MetalNativeContext,
        shape: Vec<usize>,
        dtype: DType,
        data: &[u8],
    ) -> Result<Self, MetalNativeError> {
        let mut buffer = Self::allocate(context, shape, dtype)?;
        buffer.write_from_host(data)?;
        Ok(buffer)
    }

    /// Create a buffer initialized with typed data from host
    pub fn from_vec<T>(
        context: &MetalNativeContext,
        shape: Vec<usize>,
        dtype: DType,
        data: &[T],
    ) -> Result<Self, MetalNativeError> {
        let byte_len = std::mem::size_of_val(data);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        Self::from_host(context, shape, dtype, bytes)
    }
}

// Metal buffers are safe to send between threads
unsafe impl Send for MetalNativeBuffer {}
unsafe impl Sync for MetalNativeBuffer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::NativeContext;

    #[test]
    fn test_metal_buffer_allocation() {
        if !MetalNativeContext::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalNativeContext::new().unwrap();

        // Allocate a buffer
        let shape = vec![4, 4];
        let buffer = MetalNativeBuffer::allocate(&context, shape.clone(), DType::F32);
        assert!(
            buffer.is_ok(),
            "Failed to allocate buffer: {:?}",
            buffer.err()
        );

        let buffer = buffer.unwrap();
        assert_eq!(buffer.shape(), &shape);
        assert_eq!(buffer.dtype(), DType::F32);
        assert_eq!(buffer.byte_len(), 4 * 4 * 4); // 16 floats * 4 bytes
    }

    #[test]
    fn test_metal_buffer_data_transfer() {
        if !MetalNativeContext::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalNativeContext::new().unwrap();

        // Create a buffer and write data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = MetalNativeBuffer::from_vec(&context, vec![4], DType::F32, &data).unwrap();

        // Read data back
        let result: Vec<f32> = buffer.read_vec().unwrap();
        assert_eq!(result, data);
    }
}
