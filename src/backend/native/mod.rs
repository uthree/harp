//! Native GPU API backends (without libloading)
//!
//! This module provides GPU backends that directly use Rust bindings
//! to OpenCL and Metal APIs, eliminating the need for libloading and
//! C code generation for host-side operations.
//!
//! ## Available backends
//!
//! - `opencl`: OpenCL backend using the `ocl` crate (requires `native-opencl` feature)
//! - `metal`: Metal backend using the `metal` crate (requires `native-metal` feature, macOS only)

mod traits;

#[cfg(feature = "native-opencl")]
pub mod opencl;

#[cfg(all(feature = "native-metal", target_os = "macos"))]
pub mod metal;

// Re-export traits
pub use traits::{KernelConfig, NativeBuffer, NativeCompiler, NativeContext, NativeKernel};
