//! GPU backend trait definitions
//!
//! These traits define the interface for GPU backends that directly use
//! native GPU APIs (OpenCL, Metal) through Rust bindings.

use crate::ast::DType;
use ndarray::{Array, ArrayD, Dimension as NdDimension, IxDyn};
use std::collections::HashMap;

/// Device type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceType {
    Cpu,
    IntegratedGpu,
    #[default]
    DiscreteGpu,
    Accelerator,
}

/// High-level device feature categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceFeature {
    /// Fast math operations (may sacrifice precision)
    FastMath,
    /// Half-precision (FP16) support
    HalfPrecision,
    /// Double-precision (FP64) support
    DoublePrecision,
    /// Local/shared memory support
    LocalMemory,
    /// Atomic operations support
    AtomicOperations,
    /// Subgroup/warp operations support
    SubgroupOperations,
}

/// Specific device instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceInstruction {
    /// Fused multiply-add
    Fma,
    /// Fast reciprocal square root
    Rsqrt,
    /// Atomic float add
    AtomicAddFloat,
    /// Native divide
    NativeDiv,
    /// Native exp/log
    NativeExpLog,
}

/// Operation kind for SIMD capability classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Addition
    Add,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Reciprocal
    Recip,
    /// Square root
    Sqrt,
    /// Logarithm base 2
    Log2,
    /// Exponential base 2
    Exp2,
    /// Sine
    Sin,
    /// Fused multiply-add
    Fma,
    /// Comparison operations (Lt, Le, Gt, Ge, Eq, Ne)
    Compare,
    /// Memory load
    Load,
    /// Memory store
    Store,
}

/// SIMD capability entry describing supported vector width for a specific dtype and operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimdCapability {
    /// Data type
    pub dtype: DType,
    /// Operation kind
    pub op: OpKind,
    /// Supported vector width
    pub width: usize,
}

impl SimdCapability {
    /// Create a new SIMD capability entry
    pub fn new(dtype: DType, op: OpKind, width: usize) -> Self {
        Self { dtype, op, width }
    }
}

/// Device profile containing hardware characteristics
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    /// Device type classification
    pub device_type: DeviceType,
    /// Number of compute units (CUs/SMs)
    pub compute_units: usize,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Preferred work group size range (min, max)
    pub preferred_work_group_size_range: (usize, usize),
    /// Local memory size in bytes
    pub local_memory_size: usize,
    /// Warp/wavefront size
    pub warp_size: usize,
    /// Preferred tile sizes for loop tiling
    pub preferred_tile_sizes: Vec<usize>,
    /// SIMD capabilities per dtype and operation
    pub simd_capabilities: Vec<SimdCapability>,
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self {
            device_type: DeviceType::DiscreteGpu,
            compute_units: 16,
            max_work_group_size: 1024,
            preferred_work_group_size_range: (64, 256),
            local_memory_size: 32768, // 32KB
            warp_size: 32,
            preferred_tile_sizes: vec![16, 32, 64, 128],
            simd_capabilities: Self::default_simd_capabilities(),
        }
    }
}

impl DeviceProfile {
    /// Generate default SIMD capabilities for a typical GPU
    fn default_simd_capabilities() -> Vec<SimdCapability> {
        use OpKind::*;

        let mut caps = Vec::new();

        // F32: width 4 for most operations, width 2 for division/transcendental
        for op in [Add, Mul, Fma, Compare, Load, Store] {
            caps.push(SimdCapability::new(DType::F32, op, 4));
        }
        for op in [Div, Recip, Sqrt, Log2, Exp2, Sin] {
            caps.push(SimdCapability::new(DType::F32, op, 2));
        }

        // I64: width 4 for most operations
        for op in [Add, Mul, Compare, Load, Store] {
            caps.push(SimdCapability::new(DType::I64, op, 4));
        }
        caps.push(SimdCapability::new(DType::I64, Div, 2));

        caps
    }

    /// Get the maximum SIMD width for a specific dtype and operation
    pub fn simd_width(&self, dtype: &DType, op: OpKind) -> usize {
        self.simd_capabilities
            .iter()
            .filter(|c| &c.dtype == dtype && c.op == op)
            .map(|c| c.width)
            .max()
            .unwrap_or(1) // Default to scalar if not found
    }

    /// Check if a specific SIMD width is supported for dtype and operation
    pub fn supports_simd_width(&self, dtype: &DType, op: OpKind, width: usize) -> bool {
        width <= self.simd_width(dtype, op)
    }

    /// Get all available SIMD widths for a dtype and operation (powers of 2 up to max)
    pub fn available_simd_widths(&self, dtype: &DType, op: OpKind) -> Vec<usize> {
        let max = self.simd_width(dtype, op);
        let mut widths = Vec::new();
        let mut w = 1;
        while w <= max {
            widths.push(w);
            w *= 2;
        }
        widths
    }

    /// Get the minimum SIMD width across multiple operations (for vectorizing expressions)
    pub fn common_simd_width(&self, dtype: &DType, ops: &[OpKind]) -> usize {
        ops.iter()
            .map(|&op| self.simd_width(dtype, op))
            .min()
            .unwrap_or(1)
    }

    /// Get all unique SIMD widths across all capabilities (sorted)
    pub fn all_simd_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.simd_capabilities.iter().map(|c| c.width).collect();
        widths.sort();
        widths.dedup();
        widths
    }
}

/// GPU device trait
///
/// Represents a GPU device context with hardware capability queries.
/// Concrete implementations provide their own methods for device
/// creation and management as inherent methods.
pub trait Device {
    /// Check if this backend is available on the current system
    fn is_available() -> bool;

    /// Get the device profile containing hardware characteristics
    ///
    /// Default implementation returns a generic GPU profile.
    /// Backends should override this to query actual device capabilities.
    fn profile(&self) -> DeviceProfile {
        DeviceProfile::default()
    }

    /// Check if a high-level feature is supported
    ///
    /// Default implementation returns true (conservative assumption).
    fn supports_feature(&self, _feature: DeviceFeature) -> bool {
        true
    }

    /// Check if a specific instruction is supported
    ///
    /// Default implementation returns true (conservative assumption).
    fn supports_instruction(&self, _instruction: DeviceInstruction) -> bool {
        true
    }

    /// Get SIMD width for a specific dtype and operation
    ///
    /// Returns the maximum supported vector width.
    /// Default implementation queries the device profile.
    fn simd_width(&self, dtype: &DType, op: OpKind) -> usize {
        self.profile().simd_width(dtype, op)
    }
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

    // ========================================================================
    // ndarray conversion methods
    // ========================================================================

    /// Convert buffer to ndarray with dynamic dimensions
    ///
    /// # Type Parameters
    /// * `T` - The element type (must match the buffer's dtype)
    ///
    /// # Returns
    /// An ndarray with the same shape as the buffer
    fn to_ndarray<T: Clone + 'static>(&self) -> Result<ArrayD<T>, Self::Error> {
        let data = self.read_vec::<T>()?;
        let shape = IxDyn(self.shape());
        Ok(Array::from_shape_vec(shape, data).expect("Shape mismatch in to_ndarray"))
    }

    /// Convert buffer to ndarray with static dimensions
    ///
    /// # Type Parameters
    /// * `T` - The element type (must match the buffer's dtype)
    /// * `D` - The dimension type from ndarray
    ///
    /// # Returns
    /// An ndarray with the specified dimension type
    fn to_ndarray_d<T: Clone + 'static, D: NdDimension>(&self) -> Result<Array<T, D>, Self::Error> {
        let data = self.read_vec::<T>()?;
        let shape =
            D::from_dimension(&IxDyn(self.shape())).expect("Dimension mismatch in to_ndarray_d");
        Ok(Array::from_shape_vec(shape, data).expect("Shape mismatch in to_ndarray_d"))
    }

    /// Write ndarray data to the buffer
    ///
    /// # Arguments
    /// * `array` - The ndarray to write (must have the same shape)
    fn write_ndarray<T: Clone, D: NdDimension>(
        &mut self,
        array: &Array<T, D>,
    ) -> Result<(), Self::Error> {
        // Ensure contiguous layout
        let contiguous = array
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| array.iter().cloned().collect());
        self.write_vec(&contiguous)
    }

    /// Create buffer from ndarray on the given device
    ///
    /// # Arguments
    /// * `device` - The device to allocate on
    /// * `array` - The source ndarray
    /// * `dtype` - The data type for the buffer
    fn from_ndarray<T: Clone, D: NdDimension>(
        device: &Self::Dev,
        array: &Array<T, D>,
        dtype: DType,
    ) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();
        let mut buffer = Self::allocate(device, shape, dtype)?;
        let data: Vec<T> = array
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| array.iter().cloned().collect());
        buffer.write_vec(&data)?;
        Ok(buffer)
    }
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
    pub shape_vars: HashMap<String, i64>,
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
    pub fn with_shape_var(mut self, name: impl Into<String>, value: i64) -> Self {
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
