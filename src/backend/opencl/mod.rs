//! OpenCL backend for Harp
//!
//! This module provides native GPU execution using the `ocl` crate.
//!
//! For rendering OpenCL kernel source code, see `crate::renderer::OpenCLRenderer`.

mod buffer;
mod compiler;
mod device;
mod kernel;

pub use buffer::OpenCLBuffer;
pub use compiler::OpenCLCompiler;
pub use device::{OpenCLDevice, OpenCLError};
pub use kernel::OpenCLKernel;

// Re-export renderer types for convenience
pub use crate::renderer::c_like::OptimizationLevel;
pub use crate::renderer::opencl::{OpenCLCode, OpenCLRenderer};
