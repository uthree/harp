//! Eclat: A Tensor Computation Library with JIT Compilation
//!
//! Eclat is a transpiler that generates efficient kernels for AI accelerators,
//! GPUs, and CPUs from high-level computation graphs.
//!
//! # Architecture
//!
//! Eclat provides:
//! - **ast**: AST definitions for computation graphs
//! - **graph**: Computation graph representation (includes shape module)
//! - **tensor**: High-level Tensor API with static dimension checking
//! - **grad**: Automatic differentiation (reverse-mode)
//! - **opt**: Optimization passes for AST transformations
//! - **backend**: Backend trait definitions and pipeline
//! - **lowerer**: Graph to AST lowering
//! Backend implementations are provided as separate crates:
//! - **eclat-backend-c**: C code generation backend
//! - **eclat-backend-opencl**: OpenCL GPU backend
//! - **eclat-backend-metal**: Metal GPU backend (macOS only)
//! - **eclat-viz**: Visualization tools for optimization history
//!
//! # Quick Start
//!
//! ```ignore
//! use eclat::tensor::{Tensor, D2, D1};
//! use eclat::grad::Differentiable;
//! use eclat::ast::DType;
//!
//! // Create input tensors with static dimension checking
//! let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
//! let y: Tensor<D2> = Tensor::input([32, 64], DType::F32);
//!
//! // Build computation graph (lazy evaluation)
//! let z: Tensor<D2> = &x * &y;
//! let loss: Tensor<D1> = z.sum(1);
//!
//! // Compute gradients (also lazy)
//! let grads = loss.graph().backward(&[x.graph()]);
//! ```

// ============================================================================
// Core Modules
// ============================================================================

pub mod ast;
pub mod backend;
pub mod grad;
pub mod graph;
pub mod lowerer;
pub mod opt;
pub mod tensor;

// Re-export shape module at top level for convenience
pub use graph::shape;

// ============================================================================
// Re-exports
// ============================================================================

// Core types
pub use ast::{DType, TensorDType};
pub use half::{bf16, f16};
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
        Buffer, BufferSignature, Compiler, Device, EclatDevice, Kernel, KernelSignature, Pipeline,
        Renderer,
    };

    // Shape expressions (from graph module)
    pub use crate::graph::shape::{Expr, View};

    // Tensor types
    pub use crate::tensor::{D0, D1, D2, D3, D4, D5, D6, Dimension, Dyn, Tensor};

    // Gradient computation
    pub use crate::grad::{Differentiable, GradResult, backward};
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
