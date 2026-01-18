//! CPU buffer implementation for OpenMP backend
//!
//! This module provides a CPU-based buffer using standard memory allocation.

use crate::OpenMPDevice;
use eclat::ast::DType;
use eclat::backend::traits::{Buffer, TypedBuffer};
use std::any::Any;
use std::error::Error;
use std::fmt;

/// Error type for OpenMPBuffer operations
#[derive(Debug, Clone)]
pub enum OpenMPBufferError {
    /// Size mismatch error
    SizeMismatch { expected: usize, actual: usize },
    /// Alignment error
    AlignmentError {
        buffer_size: usize,
        type_size: usize,
    },
}

impl fmt::Display for OpenMPBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenMPBufferError::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Buffer size mismatch: expected {} bytes, got {}",
                    expected, actual
                )
            }
            OpenMPBufferError::AlignmentError {
                buffer_size,
                type_size,
            } => {
                write!(
                    f,
                    "Buffer alignment error: buffer size {} is not aligned to type size {}",
                    buffer_size, type_size
                )
            }
        }
    }
}

impl Error for OpenMPBufferError {}

/// CPU buffer for OpenMP backend
///
/// Stores data in CPU memory using a Vec<u8> for raw bytes.
#[derive(Debug, Clone)]
pub struct OpenMPBuffer {
    /// Raw data storage
    data: Vec<u8>,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type of elements
    dtype: DType,
}

impl OpenMPBuffer {
    /// Get a raw pointer to the buffer data
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get a mutable raw pointer to the buffer data
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// Get the number of elements in the buffer
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

impl TypedBuffer for OpenMPBuffer {
    type Dev = OpenMPDevice;
    type Error = OpenMPBufferError;

    fn allocate(_device: &Self::Dev, shape: Vec<usize>, dtype: DType) -> Result<Self, Self::Error> {
        let num_elements: usize = shape.iter().product();
        let element_size = dtype.size_in_bytes();
        let byte_len = num_elements * element_size;

        // Allocate zeroed memory
        let data = vec![0u8; byte_len];

        Ok(Self { data, shape, dtype })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn byte_len(&self) -> usize {
        self.data.len()
    }

    fn write_from_host(&mut self, data: &[u8]) -> Result<(), Self::Error> {
        if data.len() != self.data.len() {
            return Err(Self::buffer_size_mismatch_error(
                data.len(),
                self.data.len(),
            ));
        }
        self.data.copy_from_slice(data);
        Ok(())
    }

    fn read_to_host(&self) -> Result<Vec<u8>, Self::Error> {
        Ok(self.data.clone())
    }

    fn buffer_size_mismatch_error(expected: usize, actual: usize) -> Self::Error {
        OpenMPBufferError::SizeMismatch { expected, actual }
    }

    fn buffer_alignment_error(buffer_size: usize, type_size: usize) -> Self::Error {
        OpenMPBufferError::AlignmentError {
            buffer_size,
            type_size,
        }
    }
}

impl Buffer for OpenMPBuffer {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn byte_len(&self) -> usize {
        self.data.len()
    }

    fn read_to_host(&self) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
        Ok(self.data.clone())
    }

    fn write_from_host(&mut self, data: &[u8]) -> Result<(), Box<dyn Error + Send + Sync>> {
        if data.len() != self.data.len() {
            return Err(Box::new(OpenMPBufferError::SizeMismatch {
                expected: self.data.len(),
                actual: data.len(),
            }));
        }
        self.data.copy_from_slice(data);
        Ok(())
    }

    fn clone_buffer(&self) -> Box<dyn Buffer> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_allocate() {
        let device = OpenMPDevice::new();
        let buffer = OpenMPBuffer::allocate(&device, vec![2, 3], DType::F32).unwrap();

        assert_eq!(Buffer::shape(&buffer), &[2, 3]);
        assert_eq!(Buffer::dtype(&buffer), DType::F32);
        assert_eq!(Buffer::byte_len(&buffer), 24); // 2 * 3 * 4 bytes
    }

    #[test]
    fn test_buffer_write_read() {
        let device = OpenMPDevice::new();
        let mut buffer = OpenMPBuffer::allocate(&device, vec![4], DType::F32).unwrap();

        let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let input_bytes = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, 16) };
        Buffer::write_from_host(&mut buffer, input_bytes).unwrap();

        let output_bytes = Buffer::read_to_host(&buffer).unwrap();

        let output: &[f32] =
            unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, 4) };
        assert_eq!(output, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_buffer_clone() {
        let device = OpenMPDevice::new();
        let mut buffer = OpenMPBuffer::allocate(&device, vec![2], DType::F32).unwrap();

        let input: [f32; 2] = [1.0, 2.0];
        let input_bytes = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, 8) };
        Buffer::write_from_host(&mut buffer, input_bytes).unwrap();

        let cloned = Buffer::clone_buffer(&buffer);
        assert_eq!(cloned.shape(), &[2]);
        assert_eq!(cloned.dtype(), DType::F32);

        let output_bytes = cloned.read_to_host().unwrap();
        let output: &[f32] =
            unsafe { std::slice::from_raw_parts(output_bytes.as_ptr() as *const f32, 2) };
        assert_eq!(output, &[1.0, 2.0]);
    }
}
