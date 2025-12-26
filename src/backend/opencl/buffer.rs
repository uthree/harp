//! OpenCL native buffer

use super::device::{OpenCLDevice, OpenCLError};
use crate::ast::DType;
use crate::backend::traits::{Buffer, TypedBuffer};
use ocl::{Buffer as OclBuffer, Queue, flags};
use std::sync::Arc;

/// OpenCL native buffer
///
/// Wraps an OpenCL buffer with shape and type information.
#[derive(Clone)]
pub struct OpenCLBuffer {
    buffer: Arc<OclBuffer<u8>>,
    queue: Arc<Queue>,
    shape: Vec<usize>,
    dtype: DType,
    byte_len: usize,
}

impl TypedBuffer for OpenCLBuffer {
    type Dev = OpenCLDevice;
    type Error = OpenCLError;

    fn allocate(device: &Self::Dev, shape: Vec<usize>, dtype: DType) -> Result<Self, Self::Error> {
        let element_count: usize = shape.iter().product();
        let byte_len = element_count * dtype.size_in_bytes();

        // Create OpenCL buffer
        let buffer = OclBuffer::<u8>::builder()
            .queue((*device.queue).clone())
            .flags(flags::MEM_READ_WRITE)
            .len(byte_len)
            .build()?;

        Ok(Self {
            buffer: Arc::new(buffer),
            queue: Arc::clone(&device.queue),
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

        // Get mutable access to buffer (Arc::make_mut will clone if needed)
        let buffer = Arc::make_mut(&mut self.buffer);
        buffer.write(data).enq()?;
        self.queue.finish()?;
        Ok(())
    }

    fn read_to_host(&self) -> Result<Vec<u8>, Self::Error> {
        let mut result = vec![0u8; self.byte_len];
        self.buffer.read(&mut result).enq()?;
        self.queue.finish()?;
        Ok(result)
    }

    fn buffer_size_mismatch_error(expected: usize, actual: usize) -> Self::Error {
        OpenCLError::from(format!(
            "Buffer size mismatch: expected {} bytes, got {} bytes",
            expected, actual
        ))
    }

    fn buffer_alignment_error(buffer_size: usize, type_size: usize) -> Self::Error {
        OpenCLError::from(format!(
            "Buffer size {} is not aligned to type size {}",
            buffer_size, type_size
        ))
    }
}

impl Buffer for OpenCLBuffer {
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
}

impl OpenCLBuffer {
    /// Get the underlying OpenCL buffer
    pub fn ocl_buffer(&self) -> &OclBuffer<u8> {
        &self.buffer
    }

    /// Create a buffer initialized with data from host
    pub fn from_host(
        context: &OpenCLDevice,
        shape: Vec<usize>,
        dtype: DType,
        data: &[u8],
    ) -> Result<Self, OpenCLError> {
        let mut buffer = Self::allocate(context, shape, dtype)?;
        buffer.write_from_host(data)?;
        Ok(buffer)
    }

    /// Create a buffer initialized with typed data from host
    pub fn from_vec<T>(
        context: &OpenCLDevice,
        shape: Vec<usize>,
        dtype: DType,
        data: &[T],
    ) -> Result<Self, OpenCLError> {
        let byte_len = std::mem::size_of_val(data);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        Self::from_host(context, shape, dtype, bytes)
    }
}

// OpenCL buffers are safe to send between threads
unsafe impl Send for OpenCLBuffer {}
unsafe impl Sync for OpenCLBuffer {}
