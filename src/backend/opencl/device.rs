//! OpenCL native device

use crate::ast::DType;
use crate::backend::traits::{
    Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, OpKind, SimdCapability,
};
use ocl::core::{DeviceInfo, DeviceInfoResult};
use ocl::{Context as OclContext, Device as OclDevice, Platform, Queue};
use std::sync::Arc;

/// Error type for OpenCL native operations
#[derive(Debug, Clone)]
pub struct OpenCLError(String);

impl std::fmt::Display for OpenCLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenCL error: {}", self.0)
    }
}

impl std::error::Error for OpenCLError {}

impl From<ocl::Error> for OpenCLError {
    fn from(e: ocl::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<String> for OpenCLError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for OpenCLError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// OpenCL native device
///
/// Holds the OpenCL platform, device, context, and command queue.
#[derive(Clone)]
pub struct OpenCLDevice {
    pub(crate) platform: Platform,
    pub(crate) device: OclDevice,
    pub(crate) context: Arc<OclContext>,
    pub(crate) queue: Arc<Queue>,
}

impl Device for OpenCLDevice {
    fn is_available() -> bool {
        // Check if there are any platforms with at least one device
        for platform in Platform::list() {
            if let Ok(devices) = OclDevice::list_all(platform)
                && !devices.is_empty()
            {
                return true;
            }
        }
        false
    }

    fn profile(&self) -> DeviceProfile {
        // Query device capabilities from OpenCL
        let max_work_group_size = self.device.max_wg_size().unwrap_or(1024);
        let compute_units = self.query_compute_units();
        let local_memory_size = self.query_local_mem_size();

        // Detect warp size based on vendor
        let warp_size = self.detect_warp_size();

        // Compute preferred work group size range
        let preferred_min = warp_size.max(64);
        let preferred_max = max_work_group_size.min(256);

        // Compute tile sizes based on local memory
        let preferred_tile_sizes = self.compute_preferred_tile_sizes(local_memory_size);

        // Build SIMD capabilities from detected vector widths
        let simd_capabilities = self.build_simd_capabilities();

        DeviceProfile {
            device_type: self.detect_device_type(),
            compute_units,
            max_work_group_size,
            preferred_work_group_size_range: (preferred_min, preferred_max),
            local_memory_size,
            warp_size,
            preferred_tile_sizes,
            simd_capabilities,
        }
    }

    fn supports_feature(&self, feature: DeviceFeature) -> bool {
        let extensions = self.query_extensions();
        match feature {
            DeviceFeature::FastMath => true, // Always supported in OpenCL
            DeviceFeature::HalfPrecision => extensions.contains("cl_khr_fp16"),
            DeviceFeature::DoublePrecision => extensions.contains("cl_khr_fp64"),
            DeviceFeature::LocalMemory => self.query_local_mem_size() > 0,
            DeviceFeature::AtomicOperations => true, // Basic atomics always supported
            DeviceFeature::SubgroupOperations => {
                extensions.contains("cl_khr_subgroups") || extensions.contains("cl_intel_subgroups")
            }
        }
    }

    fn supports_instruction(&self, instruction: DeviceInstruction) -> bool {
        let extensions = self.query_extensions();
        match instruction {
            DeviceInstruction::Fma => true,   // fma() is standard in OpenCL
            DeviceInstruction::Rsqrt => true, // rsqrt() is standard in OpenCL
            DeviceInstruction::AtomicAddFloat => {
                extensions.contains("cl_ext_float_atomics")
                    || extensions.contains("cl_nv_fp_atomics")
            }
            DeviceInstruction::NativeDiv => true, // native_divide() available
            DeviceInstruction::NativeExpLog => true, // native_exp/log() available
        }
    }
}

impl OpenCLDevice {
    /// Create a new context using the default device
    pub fn new() -> Result<Self, OpenCLError> {
        Self::with_device(0)
    }

    /// Query compute units from device
    fn query_compute_units(&self) -> usize {
        match self.device.info(DeviceInfo::MaxComputeUnits) {
            Ok(DeviceInfoResult::MaxComputeUnits(n)) => n as usize,
            _ => 16, // Default fallback
        }
    }

    /// Query local memory size from device
    fn query_local_mem_size(&self) -> usize {
        match self.device.info(DeviceInfo::LocalMemSize) {
            Ok(DeviceInfoResult::LocalMemSize(n)) => n as usize,
            _ => 32768, // Default 32KB fallback
        }
    }

    /// Query extensions from device
    fn query_extensions(&self) -> String {
        match self.device.info(DeviceInfo::Extensions) {
            Ok(DeviceInfoResult::Extensions(s)) => s,
            _ => String::new(),
        }
    }

    /// Query preferred vector width for float
    fn query_preferred_vector_width(&self) -> usize {
        match self.device.info(DeviceInfo::PreferredVectorWidthFloat) {
            Ok(DeviceInfoResult::PreferredVectorWidthFloat(n)) => n as usize,
            _ => 4, // Default fallback
        }
    }

    /// Query if host unified memory
    fn query_host_unified_memory(&self) -> bool {
        match self.device.info(DeviceInfo::HostUnifiedMemory) {
            Ok(DeviceInfoResult::HostUnifiedMemory(b)) => b,
            _ => false,
        }
    }

    /// Detect warp/wavefront size based on vendor
    fn detect_warp_size(&self) -> usize {
        let vendor = self.device.vendor().unwrap_or_default();
        if vendor.contains("AMD") || vendor.contains("Advanced Micro") {
            64 // AMD wavefront size
        } else {
            32 // NVIDIA warp size (also default for others)
        }
    }

    /// Compute preferred tile sizes based on local memory
    fn compute_preferred_tile_sizes(&self, local_mem_size: usize) -> Vec<usize> {
        // Calculate max tile size that fits in local memory
        // Assuming float[tile][tile] for matrix operations
        let max_tile = ((local_mem_size as f64 / 4.0).sqrt() as usize).min(128);

        [8, 16, 32, 64, 128]
            .into_iter()
            .filter(|&s| s <= max_tile)
            .collect()
    }

    /// Detect preferred vector width for F32
    fn detect_preferred_vector_width(&self) -> usize {
        let preferred = self.query_preferred_vector_width();
        match preferred {
            1 | 2 | 4 | 8 | 16 => preferred,
            _ => 4,
        }
    }

    /// Build SIMD capabilities based on device characteristics
    fn build_simd_capabilities(&self) -> Vec<SimdCapability> {
        use OpKind::*;

        let max_width = self.detect_preferred_vector_width();
        let mut caps = Vec::new();

        // F32: full width for most operations, half for division/transcendental
        let f32_width = max_width.min(4); // OpenCL float4 is common
        let f32_slow_width = (f32_width / 2).max(1);

        for op in [Add, Mul, Fma, Compare, Load, Store] {
            caps.push(SimdCapability::new(DType::F32, op, f32_width));
        }
        for op in [Div, Recip, Sqrt, Log2, Exp2, Sin] {
            caps.push(SimdCapability::new(DType::F32, op, f32_slow_width));
        }

        // Int: same width as F32 for basic ops
        for op in [Add, Mul, Compare, Load, Store] {
            caps.push(SimdCapability::new(DType::I64, op, f32_width));
        }
        caps.push(SimdCapability::new(DType::I64, Div, f32_slow_width));

        caps
    }

    /// Detect device type
    fn detect_device_type(&self) -> DeviceType {
        use ocl::DeviceType as OclDeviceType;
        match self.device.info(DeviceInfo::Type) {
            Ok(DeviceInfoResult::Type(dt)) if dt.contains(OclDeviceType::CPU) => DeviceType::Cpu,
            Ok(DeviceInfoResult::Type(dt)) if dt.contains(OclDeviceType::ACCELERATOR) => {
                DeviceType::Accelerator
            }
            Ok(DeviceInfoResult::Type(dt)) if dt.contains(OclDeviceType::GPU) => {
                // Try to detect integrated vs discrete
                // Integrated GPUs typically share memory with host
                if self.query_host_unified_memory() {
                    DeviceType::IntegratedGpu
                } else {
                    DeviceType::DiscreteGpu
                }
            }
            _ => DeviceType::DiscreteGpu,
        }
    }

    /// Create a new context for a specific device index
    pub fn with_device(device_index: usize) -> Result<Self, OpenCLError> {
        // Get default platform
        let platform = Platform::default();

        // Get all devices
        let devices = OclDevice::list_all(platform)?;
        if devices.is_empty() {
            return Err("No OpenCL devices found".into());
        }

        // Select the requested device
        let device = devices.get(device_index).cloned().ok_or_else(|| {
            format!(
                "Device index {} out of range (available: {})",
                device_index,
                devices.len()
            )
        })?;

        // Create context
        let context = OclContext::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        // Create command queue
        let queue = Queue::new(&context, device, None)?;

        Ok(Self {
            platform,
            device,
            context: Arc::new(context),
            queue: Arc::new(queue),
        })
    }

    /// Get the device name
    pub fn device_name(&self) -> String {
        self.device.name().unwrap_or_else(|_| "Unknown".to_string())
    }

    /// Get the OpenCL platform
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Get the OpenCL context
    pub fn ocl_context(&self) -> &OclContext {
        &self.context
    }

    /// Get the OpenCL command queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get the OpenCL device
    pub fn ocl_device(&self) -> OclDevice {
        self.device
    }

    /// List all available devices
    pub fn list_devices() -> Result<Vec<String>, OpenCLError> {
        let platform = Platform::default();
        let devices = OclDevice::list_all(platform)?;

        devices
            .iter()
            .map(|d| d.name().map_err(OpenCLError::from))
            .collect()
    }
}

// Safety: OpenCL context and queue are thread-safe
unsafe impl Send for OpenCLDevice {}
unsafe impl Sync for OpenCLDevice {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::backend::opencl::{OpenCLBuffer, OpenCLCompiler};
    use crate::backend::traits::{Buffer, Compiler, Device, KernelConfig};

    #[test]
    fn test_opencl_is_available() {
        // This test just checks if OpenCL is available
        let available = OpenCLDevice::is_available();
        println!("OpenCL available: {}", available);
    }

    #[test]
    fn test_opencl_context_creation() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new();
        assert!(
            context.is_ok(),
            "Failed to create context: {:?}",
            context.err()
        );

        let context = context.unwrap();
        println!("Device name: {}", context.device_name());
    }

    #[test]
    fn test_opencl_buffer_allocation() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new().unwrap();

        // Allocate a buffer
        let shape = vec![4, 4];
        let buffer = OpenCLBuffer::allocate(&context, shape.clone(), DType::F32);
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
    fn test_opencl_buffer_data_transfer() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new().unwrap();

        // Create a buffer and write data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buffer = OpenCLBuffer::from_vec(&context, vec![4], DType::F32, &data).unwrap();

        // Read data back
        let result: Vec<f32> = buffer.read_vec().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_opencl_simple_kernel() {
        if !OpenCLDevice::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let context = OpenCLDevice::new().unwrap();
        let compiler = OpenCLCompiler::new();

        // Simple kernel that adds two arrays
        let source = r#"
            __kernel void add(__global float* a, __global float* b, __global float* c) {
                int i = get_global_id(0);
                c[i] = a[i] + b[i];
            }
        "#;

        let config = KernelConfig::new("add").with_global_work_size([4, 1, 1]);

        let kernel = compiler.compile(&context, source, config);
        assert!(
            kernel.is_ok(),
            "Failed to compile kernel: {:?}",
            kernel.err()
        );

        // Create input/output buffers
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let a_buffer = OpenCLBuffer::from_vec(&context, vec![4], DType::F32, &a_data).unwrap();
        let b_buffer = OpenCLBuffer::from_vec(&context, vec![4], DType::F32, &b_data).unwrap();
        let c_buffer = OpenCLBuffer::allocate(&context, vec![4], DType::F32).unwrap();

        // Execute kernel
        let kernel = kernel.unwrap();
        let result = kernel.execute_with_buffers(&[&a_buffer, &b_buffer, &c_buffer]);
        assert!(
            result.is_ok(),
            "Failed to execute kernel: {:?}",
            result.err()
        );

        // Read result
        let c_result: Vec<f32> = c_buffer.read_vec().unwrap();
        assert_eq!(c_result, vec![6.0, 8.0, 10.0, 12.0]);
    }
}
