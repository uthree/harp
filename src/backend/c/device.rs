//! CPU device implementation for pure C backend
//!
//! This module provides a CPU-based device that does not support
//! parallel kernel execution. It is used for generating sequential
//! C code that can be compiled with any C99 compliant compiler.

use crate::ast::DType;
use crate::backend::{
    Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, OpKind, SimdCapability,
};

/// CPU device for pure C backend
///
/// This device represents a generic CPU without parallel kernel support.
/// It is used primarily for:
/// - Code generation targeting pure C
/// - Testing and debugging without GPU hardware
/// - Portability across different platforms
#[derive(Debug, Clone, Default)]
pub struct CDevice {
    /// Device name for identification
    name: String,
}

impl CDevice {
    /// Create a new CDevice with default settings
    pub fn new() -> Self {
        Self {
            name: "Generic CPU".to_string(),
        }
    }

    /// Create a CDevice with a custom name
    pub fn with_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Get the device name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Build SIMD capabilities for CPU
    ///
    /// Returns basic scalar operations (width 1) for all operations.
    fn build_simd_capabilities() -> Vec<SimdCapability> {
        use OpKind::*;

        let mut caps = Vec::new();

        // CPU supports scalar operations for all types
        for op in [Add, Mul, Div] {
            caps.push(SimdCapability::new(DType::F32, op, 1));
            caps.push(SimdCapability::new(DType::F64, op, 1));
            caps.push(SimdCapability::new(DType::I32, op, 1));
            caps.push(SimdCapability::new(DType::I64, op, 1));
        }

        caps
    }
}

impl Device for CDevice {
    fn is_available() -> bool {
        // Pure C backend is always available
        true
    }

    fn profile(&self) -> DeviceProfile {
        DeviceProfile {
            device_type: DeviceType::Cpu,
            compute_units: 1, // Single core for sequential execution
            max_work_group_size: 1,
            preferred_work_group_size_range: (1, 1),
            local_memory_size: 0, // No shared memory concept in pure C
            warp_size: 1,
            preferred_tile_sizes: vec![1, 4, 8, 16], // CPU-friendly tile sizes
            simd_capabilities: Self::build_simd_capabilities(),
        }
    }

    fn supports_feature(&self, feature: DeviceFeature) -> bool {
        match feature {
            DeviceFeature::FastMath => true,          // C has fast math options
            DeviceFeature::HalfPrecision => false,    // Standard C doesn't have half
            DeviceFeature::DoublePrecision => true,   // C supports double
            DeviceFeature::LocalMemory => false,      // No shared memory in pure C
            DeviceFeature::AtomicOperations => false, // Pure C99 doesn't have atomics
            DeviceFeature::SubgroupOperations => false, // No subgroups in pure C
            DeviceFeature::ParallelKernel => false,   // No parallel kernel execution
        }
    }

    fn supports_instruction(&self, instruction: DeviceInstruction) -> bool {
        match instruction {
            DeviceInstruction::Fma => true,    // Can use fma() from math.h
            DeviceInstruction::Rsqrt => false, // No native rsqrt in C
            DeviceInstruction::AtomicAddFloat => false, // No atomic float in pure C
            DeviceInstruction::NativeDiv => true, // Native divide available
            DeviceInstruction::NativeExpLog => true, // exp/log from math.h
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_device_creation() {
        let device = CDevice::new();
        assert_eq!(device.name(), "Generic CPU");
    }

    #[test]
    fn test_c_device_with_name() {
        let device = CDevice::with_name("Test CPU");
        assert_eq!(device.name(), "Test CPU");
    }

    #[test]
    fn test_c_device_is_available() {
        assert!(CDevice::is_available());
    }

    #[test]
    fn test_c_device_profile() {
        let device = CDevice::new();
        let profile = device.profile();

        assert_eq!(profile.device_type, DeviceType::Cpu);
        assert_eq!(profile.compute_units, 1);
        assert_eq!(profile.warp_size, 1);
    }

    #[test]
    fn test_c_device_no_parallel_kernel() {
        let device = CDevice::new();
        assert!(!device.supports_feature(DeviceFeature::ParallelKernel));
    }

    #[test]
    fn test_c_device_features() {
        let device = CDevice::new();

        // Should support
        assert!(device.supports_feature(DeviceFeature::FastMath));
        assert!(device.supports_feature(DeviceFeature::DoublePrecision));

        // Should not support
        assert!(!device.supports_feature(DeviceFeature::HalfPrecision));
        assert!(!device.supports_feature(DeviceFeature::LocalMemory));
        assert!(!device.supports_feature(DeviceFeature::AtomicOperations));
        assert!(!device.supports_feature(DeviceFeature::SubgroupOperations));
    }

    #[test]
    fn test_c_device_instructions() {
        let device = CDevice::new();

        // Should support
        assert!(device.supports_instruction(DeviceInstruction::Fma));
        assert!(device.supports_instruction(DeviceInstruction::NativeDiv));
        assert!(device.supports_instruction(DeviceInstruction::NativeExpLog));

        // Should not support
        assert!(!device.supports_instruction(DeviceInstruction::Rsqrt));
        assert!(!device.supports_instruction(DeviceInstruction::AtomicAddFloat));
    }
}
