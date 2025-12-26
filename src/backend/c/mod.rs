//! Pure C backend for Harp
//!
//! This module provides code generation for pure C99 without any
//! parallelization or external library dependencies.
//!
//! Unlike GPU backends (Metal, OpenCL), the C backend:
//! - Does not support parallel kernel execution (`ParallelKernel` feature is false)
//! - Generates sequential code suitable for any C99 compiler
//! - Has no runtime execution capability (code generation only)
//!
//! # Example
//!
//! ```
//! use harp::backend::c::CDevice;
//! use harp::backend::Device;
//! use harp::renderer::c::{CRenderer, CCode};
//! use harp::renderer::Renderer;
//!
//! // Check device capabilities
//! let device = CDevice::new();
//! assert!(CDevice::is_available());
//!
//! // Generate C code
//! let renderer = CRenderer::new();
//! // let code: CCode = renderer.render(&program);
//! ```

mod device;

pub use device::CDevice;

// Re-export renderer types for convenience
pub use crate::renderer::c::{CCode, CRenderer};
