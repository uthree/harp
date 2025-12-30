//! Harp Core: High-level Array Processor Core
//!
//! This crate provides the core functionality for Harp:
//! - AST definitions for computation graphs
//! - Optimization passes
//! - Tensor API with lazy evaluation
//! - Backend trait definitions
//!
//! Backend-specific implementations (C, Metal, OpenCL) are provided by separate crates.
//!
//! # Basic Usage
//!
//! ```ignore
//! use harp_core::prelude::*;
//! use harp_core::backend::set_device;
//!
//! // Set up a device (requires a backend crate)
//! let device = HarpDevice::auto()?;
//! set_device(device);
//!
//! // Create tensors
//! let a = Tensor::<f32, Dim2>::full([10, 20], 1.0);
//! let b = Tensor::<f32, Dim2>::full([10, 20], 2.0);
//!
//! // Lazy operations
//! let result = &a + &b;
//!
//! // Execute computation
//! result.realize().unwrap();
//! let data = result.data().unwrap();
//! ```

// Core modules
pub mod ast;
pub mod backend;
pub mod opt;
pub mod tensor;

// Optional visualization module
#[cfg(feature = "viz")]
pub mod viz;

// Re-export types
pub use ast::{DType, TensorDType};

// Re-export renderer traits
pub use backend::Renderer;

// Re-export backend traits
pub use backend::{Buffer, Compiler, Device, Kernel, KernelConfig, Pipeline};

/// Prelude module with commonly used types and traits
///
/// このモジュールをインポートすることで、Harpを使う上で必要な
/// 主要な型やトレイトを一括でインポートできます。
///
/// # Example
///
/// ```ignore
/// use harp_core::prelude::*;
/// ```
pub mod prelude {
    // Tensor types (recommended API)
    pub use crate::tensor::{
        Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension, FloatDType, Tensor,
        Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, TensorDyn,
    };

    // Data types
    pub use crate::ast::DType;

    // Backend traits
    pub use crate::backend::{
        Buffer, BufferSignature, Compiler, Device, HarpDevice, Kernel, KernelSignature, Pipeline,
        Renderer,
    };

    // Shape expressions (for advanced tensor operations)
    pub use crate::tensor::shape::{Expr, View};
}
