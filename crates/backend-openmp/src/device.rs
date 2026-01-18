//! CPU device implementation for OpenMP backend
//!
//! This module provides a CPU-based device with OpenMP parallel support.

use eclat::ast::DType;
use eclat::backend::{
    Device, DeviceFeature, DeviceInstruction, DeviceProfile, DeviceType, OpKind, SimdCapability,
};

/// CPU device for OpenMP backend
///
/// This device represents a CPU with OpenMP parallel kernel support.
#[derive(Debug, Clone, Default)]
pub struct OpenMPDevice {
    /// Device name for identification
    name: String,
    /// Number of threads (0 = auto-detect)
    num_threads: usize,
}

impl OpenMPDevice {
    /// Create a new OpenMPDevice with default settings
    pub fn new() -> Self {
        Self {
            name: "OpenMP CPU".to_string(),
            num_threads: 0, // Auto-detect
        }
    }

    /// Create an OpenMPDevice with a custom name
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            num_threads: 0,
        }
    }

    /// Set the number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Get the device name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        if self.num_threads == 0 {
            // Auto-detect: use number of CPUs
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        } else {
            self.num_threads
        }
    }

    /// Build SIMD capabilities for CPU
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

impl Device for OpenMPDevice {
    fn is_available() -> bool {
        // OpenMP backend is available if we can actually compile a simple OpenMP program
        // This checks both compiler support and omp.h availability
        use std::io::Write;

        let temp_dir = match tempfile::TempDir::new() {
            Ok(d) => d,
            Err(_) => return false,
        };

        let source_path = temp_dir.path().join("test_omp.c");
        let test_code = r#"
#include <omp.h>
int main() {
    int n = 0;
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        n += i;
    }
    return 0;
}
"#;

        if std::fs::write(&source_path, test_code).is_err() {
            return false;
        }

        let output_path = temp_dir.path().join("test_omp");

        // Try to compile with OpenMP flags
        #[cfg(target_os = "macos")]
        let args = vec![
            "-Xpreprocessor",
            "-fopenmp",
            "-lomp",
            "-o",
            output_path.to_str().unwrap_or(""),
            source_path.to_str().unwrap_or(""),
        ];

        #[cfg(not(target_os = "macos"))]
        let args = vec![
            "-fopenmp",
            "-o",
            output_path.to_str().unwrap_or(""),
            source_path.to_str().unwrap_or(""),
        ];

        std::process::Command::new("cc")
            .args(&args)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn profile(&self) -> DeviceProfile {
        let num_threads = self.num_threads();
        DeviceProfile {
            device_type: DeviceType::C, // Uses C-like code generation
            compute_units: num_threads,
            max_work_group_size: num_threads,
            preferred_work_group_size_range: (1, num_threads),
            local_memory_size: 0, // No shared memory concept
            warp_size: 1,
            preferred_tile_sizes: vec![16, 32, 64, 128], // Larger tiles for parallel execution
            simd_capabilities: Self::build_simd_capabilities(),
        }
    }

    fn supports_feature(&self, feature: DeviceFeature) -> bool {
        match feature {
            DeviceFeature::FastMath => true,
            DeviceFeature::HalfPrecision => false,
            DeviceFeature::DoublePrecision => true,
            DeviceFeature::LocalMemory => false,
            DeviceFeature::AtomicOperations => true, // OpenMP supports atomics
            DeviceFeature::SubgroupOperations => false,
            DeviceFeature::ParallelKernel => true, // OpenMP supports parallel execution
        }
    }

    fn supports_instruction(&self, instruction: DeviceInstruction) -> bool {
        match instruction {
            DeviceInstruction::Fma => true,
            DeviceInstruction::Rsqrt => false,
            DeviceInstruction::AtomicAddFloat => true, // OpenMP supports atomic float operations
            DeviceInstruction::NativeDiv => true,
            DeviceInstruction::NativeExpLog => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openmp_device_creation() {
        let device = OpenMPDevice::new();
        assert_eq!(device.name(), "OpenMP CPU");
    }

    #[test]
    fn test_openmp_device_with_name() {
        let device = OpenMPDevice::with_name("Test OpenMP");
        assert_eq!(device.name(), "Test OpenMP");
    }

    #[test]
    fn test_openmp_device_threads() {
        let device = OpenMPDevice::new().with_threads(4);
        assert_eq!(device.num_threads(), 4);
    }

    #[test]
    fn test_openmp_device_profile() {
        let device = OpenMPDevice::new();
        let profile = device.profile();

        assert_eq!(profile.device_type, DeviceType::C);
        assert!(profile.compute_units >= 1);
    }

    #[test]
    fn test_openmp_device_parallel_kernel() {
        let device = OpenMPDevice::new();
        assert!(device.supports_feature(DeviceFeature::ParallelKernel));
        assert!(device.supports_feature(DeviceFeature::AtomicOperations));
    }

    #[test]
    fn test_openmp_device_features() {
        let device = OpenMPDevice::new();

        // Should support
        assert!(device.supports_feature(DeviceFeature::FastMath));
        assert!(device.supports_feature(DeviceFeature::DoublePrecision));
        assert!(device.supports_feature(DeviceFeature::AtomicOperations));
        assert!(device.supports_feature(DeviceFeature::ParallelKernel));

        // Should not support
        assert!(!device.supports_feature(DeviceFeature::HalfPrecision));
        assert!(!device.supports_feature(DeviceFeature::LocalMemory));
        assert!(!device.supports_feature(DeviceFeature::SubgroupOperations));
    }
}
