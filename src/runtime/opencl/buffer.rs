//! OpenCL buffer implementation.

use std::any::Any;
use std::sync::Arc;

use opencl3::command_queue::CommandQueue;
use opencl3::memory::{Buffer as ClBuffer, CL_MEM_READ_WRITE};
use opencl3::types::CL_BLOCKING;

use crate::device::{Buffer, DeviceError, Result};
use crate::dtype::DType;

/// OpenCL buffer implementation.
pub struct OpenCLBuffer {
    buffer: ClBuffer<u8>,
    size: usize,
    dtype: DType,
    queue: Arc<CommandQueue>,
}

impl OpenCLBuffer {
    /// Creates a new OpenCL buffer.
    pub fn new(
        context: &opencl3::context::Context,
        queue: Arc<CommandQueue>,
        numel: usize,
        dtype: DType,
    ) -> Result<Self> {
        let size = numel * dtype.size_bytes();
        let buffer = unsafe {
            ClBuffer::create(context, CL_MEM_READ_WRITE, size, std::ptr::null_mut()).map_err(
                |e| DeviceError::BufferError(format!("Failed to create buffer: {:?}", e)),
            )?
        };

        Ok(OpenCLBuffer {
            buffer,
            size,
            dtype,
            queue,
        })
    }

    /// Creates an OpenCL buffer from existing data.
    pub fn from_data(
        context: &opencl3::context::Context,
        queue: Arc<CommandQueue>,
        data: &[u8],
        dtype: DType,
    ) -> Result<Self> {
        let size = data.len();
        let buffer = unsafe {
            ClBuffer::create(context, CL_MEM_READ_WRITE, size, std::ptr::null_mut()).map_err(
                |e| DeviceError::BufferError(format!("Failed to create buffer: {:?}", e)),
            )?
        };

        let mut opencl_buffer = OpenCLBuffer {
            buffer,
            size,
            dtype,
            queue,
        };

        opencl_buffer.copy_from_host(data);

        Ok(opencl_buffer)
    }

    /// Returns the underlying OpenCL buffer.
    pub fn cl_buffer(&self) -> &ClBuffer<u8> {
        &self.buffer
    }

    /// Returns the number of elements.
    pub fn numel(&self) -> usize {
        self.size / self.dtype.size_bytes()
    }
}

impl Buffer for OpenCLBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn copy_from_host(&mut self, data: &[u8]) {
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.buffer, CL_BLOCKING, 0, data, &[])
                .expect("Failed to write to buffer");
        }
    }

    fn copy_to_host(&self) -> Vec<u8> {
        let mut data = vec![0u8; self.size];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.buffer, CL_BLOCKING, 0, &mut data, &[])
                .expect("Failed to read from buffer");
        }
        data
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Send and Sync are safe because OpenCL handles synchronization internally
unsafe impl Send for OpenCLBuffer {}
unsafe impl Sync for OpenCLBuffer {}
