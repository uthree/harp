//! Kernel scheduling and fusion module.
//!
//! This module handles the scheduling and fusion of UOp operations
//! into optimized kernels for GPU execution.

mod analysis;
mod item;
mod kernel;
mod scheduler;

pub use analysis::GraphAnalysis;
pub use item::{FusionType, ScheduleItem};
pub use kernel::{FusedKernel, FusedOp, FusedSource, KernelInput};
pub use scheduler::Scheduler;
