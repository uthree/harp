//! Metal backend for Harp
//!
//! This module provides native GPU execution using the `metal` crate.
//!
//! For rendering Metal kernel source code, see `crate::renderer::MetalRenderer`.
//!
//! This module is only available on macOS.

mod buffer;
mod compiler;
mod device;
mod kernel;

pub use buffer::MetalBuffer;
pub use compiler::MetalCompiler;
pub use device::{MetalDevice, MetalError};
pub use kernel::MetalKernel;

// Re-export renderer types for convenience
pub use crate::renderer::c_like::OptimizationLevel;
pub use crate::renderer::metal::{MetalCode, MetalRenderer};
