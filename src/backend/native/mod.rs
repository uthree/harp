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
//!
//! ## Pipeline
//!
//! The `NativePipeline` provides end-to-end compilation from Graph to executable kernel:
//! ```ignore
//! use harp::backend::native::{NativePipeline, KernelSourceRenderer};
//! use harp::backend::native::opencl::*;
//!
//! let context = OpenCLNativeContext::new()?;
//! let compiler = OpenCLNativeCompiler::new();
//! let renderer = OpenCLRenderer::new();
//!
//! let mut pipeline = NativePipeline::new(renderer, compiler, context);
//! let kernel = pipeline.compile_graph(graph)?;
//! kernel.execute(&inputs, &mut outputs)?;
//! ```

mod pipeline;
mod traits;

#[cfg(feature = "native-opencl")]
pub mod opencl;

#[cfg(all(feature = "native-metal", target_os = "macos"))]
pub mod metal;

// Re-export traits
pub use pipeline::{
    CompiledNativeKernel, KernelSourceRenderer, NativeOptimizationHistories, NativePipeline,
    NativePipelineConfig,
};
pub use traits::{KernelConfig, NativeBuffer, NativeCompiler, NativeContext, NativeKernel};
