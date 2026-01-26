//! Lowerer module for converting computation graphs to AST
//!
//! This module provides functionality to lower `GraphNode` computation graphs
//! into executable AST (`AstNode`) representations.
//!
//! # Overview
//!
//! The lowering process consists of several stages:
//! 1. **Fusion**: Combine compatible operations (view fusion, elementwise+reduce fusion)
//! 2. **Index Generation**: Convert Views to index expressions
//! 3. **Loop Generation**: Create loop structures for parallel execution
//! 4. **Lowering**: Generate final AST kernels
//!
//! # Strategy: 1 Node = 1 Kernel
//!
//! After fusion, each remaining GraphNode is lowered to exactly one kernel.
//! This simplifies buffer management and provides clear execution boundaries.

mod fusion;
mod index_gen;
mod loop_gen;
mod lower;

#[cfg(test)]
mod tests;

// Re-exports
pub use fusion::{FusionPass, fuse_elementwise_reduce, fuse_views};
pub use index_gen::IndexGenerator;
pub use loop_gen::LoopGenerator;
pub use lower::{Lowerer, LoweringError, LoweringResult};
