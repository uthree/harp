//! GPU backend trait definitions
//!
//! These traits define the interface for GPU backends that directly use
//! native GPU APIs (OpenCL, Metal) through Rust bindings.

use crate::ast::DType;
use std::collections::HashMap;

/// GPU device marker trait
///
/// Marks types that represent a GPU device context.
/// Concrete implementations provide their own methods for device
/// creation and management as inherent methods.
pub trait Device {
    /// Check if this backend is available on the current system
    fn is_available() -> bool;
}

/// GPU buffer
///
/// Represents a memory buffer on the GPU device.
pub trait Buffer: Sized + Clone + Send + Sync {
    type Dev: Device;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Allocate a new buffer on the device
    fn allocate(device: &Self::Dev, shape: Vec<usize>, dtype: DType) -> Result<Self, Self::Error>;

    /// Get the shape of the buffer
    fn shape(&self) -> &[usize];

    /// Get the data type of the buffer elements
    fn dtype(&self) -> DType;

    /// Get the total number of bytes in the buffer
    fn byte_len(&self) -> usize;

    /// Write data from host memory to the device buffer
    fn write_from_host(&mut self, data: &[u8]) -> Result<(), Self::Error>;

    /// Read data from the device buffer to host memory
    fn read_to_host(&self) -> Result<Vec<u8>, Self::Error>;

    /// Write typed data from host memory to the device buffer
    fn write_vec<T>(&mut self, data: &[T]) -> Result<(), Self::Error> {
        let byte_len = std::mem::size_of_val(data);

        if byte_len != self.byte_len() {
            return Err(Self::buffer_size_mismatch_error(byte_len, self.byte_len()));
        }

        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        self.write_from_host(bytes)
    }

    /// Read typed data from the device buffer to host memory
    fn read_vec<T: Clone + 'static>(&self) -> Result<Vec<T>, Self::Error> {
        let bytes = self.read_to_host()?;
        let type_size = std::mem::size_of::<T>();

        if bytes.len() % type_size != 0 {
            return Err(Self::buffer_alignment_error(bytes.len(), type_size));
        }

        let len = bytes.len() / type_size;
        let mut result = Vec::with_capacity(len);

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const T, result.as_mut_ptr(), len);
            result.set_len(len);
        }

        Ok(result)
    }

    /// Create a size mismatch error (for default implementation)
    fn buffer_size_mismatch_error(expected: usize, actual: usize) -> Self::Error;

    /// Create an alignment error (for default implementation)
    fn buffer_alignment_error(buffer_size: usize, type_size: usize) -> Self::Error;
}

/// Kernel execution configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Entry point function name in the kernel source
    pub entry_point: String,
    /// Global work size (grid dimensions)
    pub global_work_size: [usize; 3],
    /// Local work size (threadgroup dimensions)
    pub local_work_size: Option<[usize; 3]>,
    /// Shape variables for dynamic shapes
    pub shape_vars: HashMap<String, isize>,
}

impl KernelConfig {
    /// Create a new kernel configuration
    pub fn new(entry_point: impl Into<String>) -> Self {
        Self {
            entry_point: entry_point.into(),
            global_work_size: [1, 1, 1],
            local_work_size: None,
            shape_vars: HashMap::new(),
        }
    }

    /// Set the global work size
    pub fn with_global_work_size(mut self, size: [usize; 3]) -> Self {
        self.global_work_size = size;
        self
    }

    /// Set the local work size
    pub fn with_local_work_size(mut self, size: [usize; 3]) -> Self {
        self.local_work_size = Some(size);
        self
    }

    /// Add a shape variable
    pub fn with_shape_var(mut self, name: impl Into<String>, value: isize) -> Self {
        self.shape_vars.insert(name.into(), value);
        self
    }
}

/// GPU kernel
///
/// Represents a compiled compute kernel that can be executed on the GPU.
pub trait Kernel: Sized + Clone + Send + Sync {
    type Buffer: Buffer;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Get the kernel configuration
    fn config(&self) -> &KernelConfig;

    /// Execute the kernel with the given input and output buffers
    ///
    /// Buffers are passed in order: inputs first, then outputs.
    /// The order must match the kernel's argument order.
    fn execute(
        &self,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
    ) -> Result<(), Self::Error>;

    /// Execute the kernel with explicit grid and local sizes
    ///
    /// This allows overriding the default dispatch sizes stored in the kernel config.
    /// Useful for multi-kernel programs where each invocation may have different sizes.
    ///
    /// # Arguments
    /// * `inputs` - Input buffer references
    /// * `outputs` - Output buffer references (mutable)
    /// * `grid_size` - Global work size (total threads to dispatch)
    /// * `local_size` - Local work size (threads per group)
    ///
    /// Default implementation ignores the size parameters and calls `execute`.
    /// Backends should override this to support dynamic dispatch sizes.
    fn execute_with_sizes(
        &self,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
        _grid_size: [usize; 3],
        _local_size: [usize; 3],
    ) -> Result<(), Self::Error> {
        // Default implementation: ignore sizes and use config's sizes
        self.execute(inputs, outputs)
    }
}

/// GPU kernel compiler
///
/// Compiles kernel source code into executable kernels.
pub trait Compiler: Sized {
    type Dev: Device;
    type Kernel: Kernel;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create a new compiler
    fn new() -> Self;

    /// Compile kernel source code into an executable kernel
    fn compile(
        &self,
        device: &Self::Dev,
        source: &str,
        config: KernelConfig,
    ) -> Result<Self::Kernel, Self::Error>;
}
