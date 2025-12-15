//! Metal native backend
//!
//! This module provides a Metal backend using the `metal` crate,
//! eliminating the need for libloading and Objective-C++ host code generation.

mod buffer;
mod compiler;
mod context;
mod kernel;

pub use buffer::MetalNativeBuffer;
pub use compiler::MetalNativeCompiler;
pub use context::{MetalNativeContext, MetalNativeError};
pub use kernel::MetalNativeKernel;
