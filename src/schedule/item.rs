//! Schedule item definitions.

use crate::uop::UOp;

/// Type of kernel fusion applied to a schedule item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionType {
    /// No fusion - single operation
    Single,
    /// Elementwise operations fused together
    Elementwise,
    /// Reduce operation with optional elementwise pre-processing
    Reduce,
}

/// A schedule item represents one kernel execution in the schedule.
/// It may contain multiple fused UOps that will be executed as a single kernel.
#[derive(Debug, Clone)]
pub struct ScheduleItem {
    /// The output UOp for this schedule item
    pub output: UOp,
    /// All UOps fused into this schedule item (in execution order)
    pub fused_ops: Vec<UOp>,
    /// Indices into the global input list for this kernel
    pub inputs: Vec<usize>,
    /// The type of fusion applied
    pub fusion_type: FusionType,
}

impl ScheduleItem {
    /// Create a new schedule item with a single operation (no fusion).
    pub fn single(output: UOp, inputs: Vec<usize>) -> Self {
        Self {
            output: output.clone(),
            fused_ops: vec![output],
            inputs,
            fusion_type: FusionType::Single,
        }
    }

    /// Create a new schedule item with fused elementwise operations.
    pub fn elementwise(output: UOp, fused_ops: Vec<UOp>, inputs: Vec<usize>) -> Self {
        Self {
            output,
            fused_ops,
            inputs,
            fusion_type: FusionType::Elementwise,
        }
    }

    /// Create a new schedule item for a reduce operation.
    pub fn reduce(output: UOp, fused_ops: Vec<UOp>, inputs: Vec<usize>) -> Self {
        Self {
            output,
            fused_ops,
            inputs,
            fusion_type: FusionType::Reduce,
        }
    }

    /// Returns true if this schedule item has fused operations.
    pub fn is_fused(&self) -> bool {
        self.fused_ops.len() > 1
    }
}
