//! OpenCL native backend
//!
//! This module provides an OpenCL backend using the `ocl` crate,
//! eliminating the need for libloading and C host code generation.

mod buffer;
mod compiler;
mod context;
mod kernel;

pub use buffer::OpenCLNativeBuffer;
pub use compiler::OpenCLNativeCompiler;
pub use context::OpenCLNativeContext;
pub use kernel::OpenCLNativeKernel;
