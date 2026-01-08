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
//! - **viz**: Visualization tools (optional, feature: viz)
//!
//! Backend implementations are provided as separate crates:
//! - **harp-backend-c**: C code generation backend
//! - **harp-backend-opencl**: OpenCL GPU backend
//! - **harp-backend-metal**: Metal GPU backend (macOS only)
//!
//! # Feature Flags
//!
//! - `viz`: Enable visualization tools

// ============================================================================
// Core Modules
// ============================================================================

pub mod ast;
pub mod backend;
pub mod opt;
pub mod shape;

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
// Renderer re-exports
// ============================================================================

/// Renderer types
pub mod renderer {
    // Core renderer types are always available
    pub use crate::backend::renderer::{
        CLikeRenderer, GenericRenderer, OptimizationLevel, Renderer, extract_buffer_placeholders,
    };
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
}
