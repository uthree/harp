//! # Performance Optimization Modules
//!
//! This module groups together various components of the library that are
//! responsible for optimizing the computation graph, both at the `Tensor`
//! level and the `UOp` level.

pub mod autotuner;
pub mod linearizer;
pub mod optimizer;
pub mod pattern;
