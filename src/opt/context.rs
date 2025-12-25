//! Optimization context for hardware-aware optimization
//!
//! This module provides a context that carries device information
//! throughout the optimization pipeline.

use crate::ast::DType;
use crate::backend::{Device, DeviceFeature, DeviceInstruction, DeviceProfile, OpKind};
use std::collections::HashSet;

/// Optimization context carrying device information
///
/// This context is passed to suggesters and cost estimators to enable
/// hardware-aware optimization decisions.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device profile with hardware characteristics
    pub profile: DeviceProfile,
    /// Set of supported features
    pub features: HashSet<DeviceFeature>,
    /// Set of supported instructions
    pub instructions: HashSet<DeviceInstruction>,
}

impl DeviceCapabilities {
    /// Create a context from a device
    pub fn from_device<D: Device>(device: &D) -> Self {
        let profile = device.profile();

        // Collect supported features
        let features = [
            DeviceFeature::FastMath,
            DeviceFeature::HalfPrecision,
            DeviceFeature::DoublePrecision,
            DeviceFeature::LocalMemory,
            DeviceFeature::AtomicOperations,
            DeviceFeature::SubgroupOperations,
        ]
        .into_iter()
        .filter(|&f| device.supports_feature(f))
        .collect();

        // Collect supported instructions
        let instructions = [
            DeviceInstruction::Fma,
            DeviceInstruction::Rsqrt,
            DeviceInstruction::AtomicAddFloat,
            DeviceInstruction::NativeDiv,
            DeviceInstruction::NativeExpLog,
        ]
        .into_iter()
        .filter(|&i| device.supports_instruction(i))
        .collect();

        Self {
            profile,
            features,
            instructions,
        }
    }

    /// Create a default GPU context (for backward compatibility)
    pub fn default_gpu() -> Self {
        Self {
            profile: DeviceProfile::default(),
            features: [
                DeviceFeature::FastMath,
                DeviceFeature::HalfPrecision,
                DeviceFeature::DoublePrecision,
                DeviceFeature::LocalMemory,
                DeviceFeature::AtomicOperations,
                DeviceFeature::SubgroupOperations,
            ]
            .into_iter()
            .collect(),
            instructions: [
                DeviceInstruction::Fma,
                DeviceInstruction::Rsqrt,
                DeviceInstruction::AtomicAddFloat,
                DeviceInstruction::NativeDiv,
                DeviceInstruction::NativeExpLog,
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Check if a feature is supported
    pub fn supports_feature(&self, feature: DeviceFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Check if an instruction is supported
    pub fn supports_instruction(&self, instruction: DeviceInstruction) -> bool {
        self.instructions.contains(&instruction)
    }

    /// Get preferred tile sizes
    pub fn preferred_tile_sizes(&self) -> &[usize] {
        &self.profile.preferred_tile_sizes
    }

    /// Get preferred work group size range
    pub fn preferred_work_group_size_range(&self) -> (usize, usize) {
        self.profile.preferred_work_group_size_range
    }

    /// Get maximum work group size
    pub fn max_work_group_size(&self) -> usize {
        self.profile.max_work_group_size
    }

    /// Get SIMD width for a specific dtype and operation
    pub fn simd_width(&self, dtype: &DType, op: OpKind) -> usize {
        self.profile.simd_width(dtype, op)
    }

    /// Get common SIMD width across multiple operations
    pub fn common_simd_width(&self, dtype: &DType, ops: &[OpKind]) -> usize {
        self.profile.common_simd_width(dtype, ops)
    }

    /// Get available SIMD widths for a dtype and operation
    pub fn available_simd_widths(&self, dtype: &DType, op: OpKind) -> Vec<usize> {
        self.profile.available_simd_widths(dtype, op)
    }

    /// Get all unique SIMD widths across all capabilities
    pub fn all_simd_widths(&self) -> Vec<usize> {
        self.profile.all_simd_widths()
    }
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self::default_gpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_context() {
        let ctx = DeviceCapabilities::default_gpu();
        assert!(ctx.supports_feature(DeviceFeature::FastMath));
        assert!(ctx.supports_instruction(DeviceInstruction::Fma));
        assert_eq!(ctx.max_work_group_size(), 1024);
    }

    #[test]
    fn test_preferred_tile_sizes() {
        let ctx = DeviceCapabilities::default_gpu();
        let tiles = ctx.preferred_tile_sizes();
        assert!(!tiles.is_empty());
        assert!(tiles.contains(&32));
    }

    #[test]
    fn test_work_group_size_range() {
        let ctx = DeviceCapabilities::default_gpu();
        let (min, max) = ctx.preferred_work_group_size_range();
        assert!(min <= max);
        assert!(max <= ctx.max_work_group_size());
    }
}
