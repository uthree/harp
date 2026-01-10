//! Tensor module with type-safe dimension tracking
//!
//! This module provides a high-level `Tensor<D>` API that wraps the lower-level
//! computation graph (`GraphNode`) with compile-time dimension checking.
//!
//! # Features
//!
//! - **Static dimension checking**: Dimension mismatches are caught at compile time
//! - **Lazy evaluation**: Operations build a computation graph, executed with `realize()`
//! - **Operator overloading**: Natural mathematical syntax (`+`, `-`, `*`, `/`)
//!
//! # Example
//!
//! ```ignore
//! use eclat::tensor::{Tensor, D2, D1};
//! use eclat::ast::DType;
//!
//! // Create input tensors
//! let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
//! let y: Tensor<D2> = Tensor::input([32, 64], DType::F32);
//!
//! // Build computation graph (lazy)
//! let z: Tensor<D2> = &x * &y;
//! let loss: Tensor<D1> = z.sum(1);
//!
//! // Type errors are caught at compile time:
//! // let bad: Tensor<D3> = x.sum(1);  // Error: expected D3, found D1
//! ```

mod autograd;
pub mod dim;
mod ops;
mod realize;
mod tensor;

// Re-export dimension types
pub use dim::{D0, D1, D2, D3, D4, D5, D6, DimAdd1, DimEq, DimSub1, Dimension, Dyn};

// Re-export tensor type
pub use tensor::Tensor;

// Re-export autograd types
pub use autograd::BackwardError;
