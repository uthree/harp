//! CPU device implementation for Rust backend
//!
//! This module provides a CPU-based device that compiles generated Rust code
//! into a shared library for runtime execution.

use eclat::ast::DType;
use eclat::backend::{
    Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, OpKind, SimdCapability,
};

/// CPU device for Rust backend
///
/// This device represents a generic CPU that compiles Rust code at runtime.
/// It is used primarily for:
/// - Learning/experimental purposes
/// - Testing Rust code generation
/// - Portability across different platforms
#[derive(Debug, Clone, Default)]
pub struct RustDevice {
    /// Device name for identification
    name: String,
}

impl RustDevice {
    /// Create a new RustDevice with default settings
    pub fn new() -> Self {
        Self {
            name: "Rust CPU".to_string(),
        }
    }

    /// Create a RustDevice with a custom name
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

impl Device for RustDevice {
    fn is_available() -> bool {
        // Check if rustc is available
        std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn profile(&self) -> DeviceProfile {
        DeviceProfile {
            device_type: DeviceType::C, // Use C device type (CPU-based)
            compute_units: 1,           // Single core for sequential execution
            max_work_group_size: 1,
            preferred_work_group_size_range: (1, 1),
            local_memory_size: 0, // No shared memory concept
            warp_size: 1,
            preferred_tile_sizes: vec![1, 4, 8, 16], // CPU-friendly tile sizes
            simd_capabilities: Self::build_simd_capabilities(),
            matrix_capabilities: Vec::new(), // No matrix operations in Rust backend
        }
    }

    fn supports_feature(&self, feature: DeviceFeature) -> bool {
        match feature {
            DeviceFeature::FastMath => true, // Rust has fast math intrinsics
            DeviceFeature::HalfPrecision => false, // No native half in std Rust
            DeviceFeature::DoublePrecision => true, // Rust supports f64
            DeviceFeature::LocalMemory => false, // No shared memory
            DeviceFeature::AtomicOperations => false, // Sequential execution
            DeviceFeature::SubgroupOperations => false, // No subgroups
            DeviceFeature::ParallelKernel => false, // No parallel kernel execution
            DeviceFeature::MatrixOperations => false, // No matrix operations in Rust backend
        }
    }

    fn supports_instruction(&self, instruction: DeviceInstruction) -> bool {
        match instruction {
            DeviceInstruction::Fma => true,    // f32::mul_add / f64::mul_add
            DeviceInstruction::Rsqrt => false, // No native rsqrt in std Rust
            DeviceInstruction::AtomicAddFloat => false, // Sequential execution
            DeviceInstruction::NativeDiv => true, // Native divide available
            DeviceInstruction::NativeExpLog => true, // exp/ln methods available
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_device_creation() {
        let device = RustDevice::new();
        assert_eq!(device.name(), "Rust CPU");
    }

    #[test]
    fn test_rust_device_with_name() {
        let device = RustDevice::with_name("Test Rust CPU");
        assert_eq!(device.name(), "Test Rust CPU");
    }

    #[test]
    fn test_rust_device_is_available() {
        // This should pass if rustc is installed
        let available = RustDevice::is_available();
        println!("Rust device available: {}", available);
        // Don't assert - rustc might not be installed in all environments
    }

    #[test]
    fn test_rust_device_profile() {
        let device = RustDevice::new();
        let profile = device.profile();

        assert_eq!(profile.device_type, DeviceType::C);
        assert_eq!(profile.compute_units, 1);
        assert_eq!(profile.warp_size, 1);
    }

    #[test]
    fn test_rust_device_no_parallel_kernel() {
        let device = RustDevice::new();
        assert!(!device.supports_feature(DeviceFeature::ParallelKernel));
    }

    #[test]
    fn test_rust_device_features() {
        let device = RustDevice::new();

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
    fn test_rust_device_instructions() {
        let device = RustDevice::new();

        // Should support
        assert!(device.supports_instruction(DeviceInstruction::Fma));
        assert!(device.supports_instruction(DeviceInstruction::NativeDiv));
        assert!(device.supports_instruction(DeviceInstruction::NativeExpLog));

        // Should not support
        assert!(!device.supports_instruction(DeviceInstruction::Rsqrt));
        assert!(!device.supports_instruction(DeviceInstruction::AtomicAddFloat));
    }
}
