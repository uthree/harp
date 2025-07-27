//! # Performance Optimization Modules
//!
//! This module groups together various components of the library that are
//! responsible for optimizing the computation graph, both at the `Tensor`
//! level and the `UOp` level.

pub mod autotuner;
pub mod linearizer;
pub mod optimizer;
pub mod pattern;

pub use autotuner::{
    Autotuner, BackendOptions, Configuration, GridSearch, OptimizationRule, SearchSpace,
};
pub use linearizer::Linearizer;
pub use optimizer::Optimizer;
pub use pattern::{PatternMatcher, TPat, TPatRule, TensorPatternMatcher, UPat};
