//! Harp: High-level Array Processor
//!
//! Harp is a transpiler that generates efficient kernels for AI accelerators,
//! GPUs, and CPUs from high-level computation graphs.
//!
//! # Architecture
//!
//! Harp is organized into several crates:
//!
//! - **harp-core**: Core functionality (AST, optimization, Tensor API, backend traits)
//! - **harp-backend-c**: C code generation backend
//! - **harp-backend-metal**: Metal backend (macOS only)
//! - **harp-backend-opencl**: OpenCL backend
//! - **harp** (this crate): Facade that re-exports everything
//!
//! # Quick Start
//!
//! ```ignore
//! use harp::prelude::*;
//!
//! // Set up a device (auto-selects best available)
//! let device = HarpDevice::auto()?;
//! device.set_as_default();
//!
//! // Create tensors
//! let a = Tensor::<f32, Dim2>::full([10, 20], 1.0);
//! let b = Tensor::<f32, Dim2>::full([10, 20], 2.0);
//!
//! // Lazy operations
//! let result = &a + &b;
//!
//! // Execute computation
//! result.realize()?;
//! let data = result.data().unwrap();
//! ```
//!
//! # Feature Flags
//!
//! - `c`: Enable C code generation backend
//! - `opencl`: Enable OpenCL GPU backend
//! - `metal`: Enable Metal GPU backend (macOS only)
//! - `viz`: Enable visualization tools

// ============================================================================
// Backend Initialization
// ============================================================================

// Automatically initialize backends when the crate is loaded

#[cfg(feature = "c")]
#[ctor::ctor]
fn init_c() {
    harp_backend_c::init();
}

#[cfg(feature = "opencl")]
#[ctor::ctor]
fn init_opencl() {
    harp_backend_opencl::init();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[ctor::ctor]
fn init_metal() {
    harp_backend_metal::init();
}

// ============================================================================
// Re-exports from harp-core
// ============================================================================

// Core modules
pub use harp_core::ast;
pub use harp_core::backend;
pub use harp_core::opt;
pub use harp_core::tensor;

// Visualization (optional)
#[cfg(feature = "viz")]
pub use harp_core::viz;

// Core types
pub use harp_core::{Buffer, Compiler, Device, Kernel, KernelConfig, Pipeline, Renderer};
pub use harp_core::{DType, TensorDType};

// Prelude
pub use harp_core::prelude;

// ============================================================================
// Re-exports from backend crates
// ============================================================================

/// C backend types (when enabled)
#[cfg(feature = "c")]
pub mod c {
    pub use harp_backend_c::*;
}

/// OpenCL backend types (when enabled)
#[cfg(feature = "opencl")]
pub mod opencl {
    pub use harp_backend_opencl::*;
}

/// Metal backend types (when enabled)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal {
    pub use harp_backend_metal::*;
}

// ============================================================================
// Renderer re-exports (for convenience)
// ============================================================================

/// Renderer types from enabled backends
pub mod renderer {
    // Core renderer types are always available
    pub use harp_core::backend::renderer::{
        CLikeRenderer, GenericRenderer, OptimizationLevel, Renderer, extract_buffer_placeholders,
    };

    // C renderer
    #[cfg(feature = "c")]
    pub use harp_backend_c::renderer::{CCode, CRenderer};

    // OpenCL renderer
    #[cfg(feature = "opencl")]
    pub use harp_backend_opencl::renderer::{OpenCLCode, OpenCLRenderer};

    // Metal renderer
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub use harp_backend_metal::renderer::{MetalCode, MetalRenderer};
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
        let _ = Tensor::<f32, Dim2>::zeros([2, 3]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_opencl_backend_available() {
        use super::backend::HarpDevice;
        // Just check that the function doesn't panic
        let _ = HarpDevice::list_all();
    }
}
