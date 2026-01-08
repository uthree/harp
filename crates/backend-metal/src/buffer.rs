//! Metal native buffer

use super::device::{MetalDevice, MetalError};
use harp::ast::DType;
use harp::backend::traits::{Buffer, TypedBuffer};
use metal::{Buffer as MtlBuffer, MTLResourceOptions};
use std::sync::Arc;

/// Metal native buffer
///
/// Wraps a Metal buffer with shape and type information.
#[derive(Clone)]
pub struct MetalBuffer {
    buffer: Arc<MtlBuffer>,
    shape: Vec<usize>,
    dtype: DType,
    byte_len: usize,
}

impl TypedBuffer for MetalBuffer {
    type Dev = MetalDevice;
    type Error = MetalError;

    fn allocate(device: &Self::Dev, shape: Vec<usize>, dtype: DType) -> Result<Self, Self::Error> {
        let element_count: usize = shape.iter().product();
        let byte_len = element_count * dtype.size_in_bytes();

        // Create Metal buffer with shared storage mode for CPU/GPU access
        let buffer = device
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
        MetalError::from(format!(
            "Buffer size mismatch: expected {} bytes, got {} bytes",
            expected, actual
        ))
    }

    fn buffer_alignment_error(buffer_size: usize, type_size: usize) -> Self::Error {
        MetalError::from(format!(
            "Buffer size {} is not aligned to type size {}",
            buffer_size, type_size
        ))
    }
}

impl Buffer for MetalBuffer {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn read_to_host(&self) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        TypedBuffer::read_to_host(self)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn write_from_host(
        &mut self,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        TypedBuffer::write_from_host(self, data)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn clone_buffer(&self) -> Box<dyn Buffer> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl MetalBuffer {
    /// Get the underlying Metal buffer
    pub fn mtl_buffer(&self) -> &MtlBuffer {
        &self.buffer
    }

    /// Create a buffer initialized with data from host
    pub fn from_host(
        context: &MetalDevice,
        shape: Vec<usize>,
        dtype: DType,
        data: &[u8],
    ) -> Result<Self, MetalError> {
        let mut buffer = Self::allocate(context, shape, dtype)?;
        TypedBuffer::write_from_host(&mut buffer, data)?;
        Ok(buffer)
    }

    /// Create a buffer initialized with typed data from host
    pub fn from_vec<T>(
        context: &MetalDevice,
        shape: Vec<usize>,
        dtype: DType,
        data: &[T],
    ) -> Result<Self, MetalError> {
        let byte_len = std::mem::size_of_val(data);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        Self::from_host(context, shape, dtype, bytes)
    }
}

// Metal buffers are safe to send between threads
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

#[cfg(test)]
mod tests {
    use super::*;
    use harp::backend::traits::{Device, TypedBuffer};

    #[test]
    fn test_metal_buffer_allocation() {
        if !MetalDevice::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalDevice::new().unwrap();

        // Allocate a buffer
        let shape = vec![4, 4];
        let buffer = MetalBuffer::allocate(&context, shape.clone(), DType::F32);
        assert!(
            buffer.is_ok(),
            "Failed to allocate buffer: {:?}",
            buffer.err()
        );

        let buffer = buffer.unwrap();
        assert_eq!(TypedBuffer::shape(&buffer), &shape);
        assert_eq!(TypedBuffer::dtype(&buffer), DType::F32);
        assert_eq!(TypedBuffer::byte_len(&buffer), 4 * 4 * 4); // 16 floats * 4 bytes
    }

    #[test]
    fn test_metal_buffer_data_transfer() {
        if !MetalDevice::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalDevice::new().unwrap();

        // Create a buffer and write data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = MetalBuffer::from_vec(&context, vec![4], DType::F32, &data).unwrap();

        // Read data back
        let result: Vec<f32> = buffer.read_vec().unwrap();
        assert_eq!(result, data);
    }
}
