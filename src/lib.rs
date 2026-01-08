//! Harp: High-level Array Processor
//!
//! Harp is a transpiler that generates efficient kernels for AI accelerators,
//! GPUs, and CPUs from high-level computation graphs.
//!
//! # Architecture
//!
//! Harp provides:
//! - **ast**: AST definitions for computation graphs
//! - **opt**: Optimization passes for AST transformations
//! - **shape**: Shape expressions for tensor operations
//! - **backend**: Backend trait definitions and pipeline
//! - **backends**: Backend implementations (C, OpenCL, Metal)
//! - **viz**: Visualization tools (optional, feature: viz)
//!
//! # Feature Flags
//!
//! - `c`: Enable C code generation backend
//! - `opencl`: Enable OpenCL GPU backend
//! - `metal`: Enable Metal GPU backend (macOS only)
//! - `viz`: Enable visualization tools

// ============================================================================
// Core Modules
// ============================================================================

pub mod ast;
pub mod backend;
pub mod opt;
pub mod shape;

// Backend implementations (feature-gated)
pub mod backends;

// Optional visualization module
#[cfg(feature = "viz")]
pub mod viz;

// ============================================================================
// Re-exports
// ============================================================================

// Core types
pub use ast::{DType, TensorDType};
pub use backend::{Buffer, Compiler, Device, Kernel, KernelConfig, Pipeline, Renderer};

// ============================================================================
// Prelude
// ============================================================================

/// Prelude module with commonly used types and traits
pub mod prelude {
    // Data types
    pub use crate::ast::DType;

    // Backend traits
    pub use crate::backend::{
        Buffer, BufferSignature, Compiler, Device, HarpDevice, Kernel, KernelSignature, Pipeline,
        Renderer,
    };

    // Shape expressions
    pub use crate::shape::{Expr, View};
}

// ============================================================================
// Backend Initialization
// ============================================================================

#[cfg(feature = "c")]
#[ctor::ctor]
fn init_c() {
    backends::c::init();
}

#[cfg(feature = "opencl")]
#[ctor::ctor]
fn init_opencl() {
    backends::opencl::init();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[ctor::ctor]
fn init_metal() {
    backends::metal::init();
}

// ============================================================================
// Backend Re-exports (for convenience)
// ============================================================================

/// C backend types (when enabled)
#[cfg(feature = "c")]
pub mod c {
    pub use crate::backends::c::*;
}

/// OpenCL backend types (when enabled)
#[cfg(feature = "opencl")]
pub mod opencl {
    pub use crate::backends::opencl::*;
}

/// Metal backend types (when enabled)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal {
    pub use crate::backends::metal::*;
}

// ============================================================================
// Renderer re-exports
// ============================================================================

/// Renderer types from enabled backends
pub mod renderer {
    // Core renderer types are always available
    pub use crate::backend::renderer::{
        CLikeRenderer, GenericRenderer, OptimizationLevel, Renderer, extract_buffer_placeholders,
    };

    // C renderer
    #[cfg(feature = "c")]
    pub use crate::backends::c::renderer::{CCode, CRenderer};

    // OpenCL renderer
    #[cfg(feature = "opencl")]
    pub use crate::backends::opencl::renderer::{OpenCLCode, OpenCLRenderer};

    // Metal renderer
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub use crate::backends::metal::renderer::{MetalCode, MetalRenderer};
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn test_facade_compiles() {
        // Verify that the facade compiles correctly
        use super::prelude::*;
        let _ = Expr::Const(42);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_opencl_backend_available() {
        use super::backend::HarpDevice;
        // Just check that the function doesn't panic
        let _ = HarpDevice::list_all();
    }
}
