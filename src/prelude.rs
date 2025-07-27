//! The `harp` prelude.
//!
//! This module re-exports the most commonly used types, traits, and functions
//! from the `harp` library. It is designed to be glob-imported for convenience.
//!
//! By importing everything from this module, you can easily access all the essential
//! components needed to build and execute computation graphs.
//!
//! # Example
//!
//! ```
//! use harp::prelude::*;
//!
//! // Now you can use essential types like `Tensor`, `ClangBackend`, `DType`,
//! // and traits like `Backend` and `ToDot` directly.
//! ```

pub use crate::backends::{Backend, Buffer, ClangBackend};
pub use crate::context::backend;
pub use crate::dot::ToDot;
pub use crate::dtype::DType;
pub use crate::lowerizer::Lowerizer;
pub use crate::shapetracker::ShapeTracker;
pub use crate::tensor::{Tensor, TensorOp};
pub use crate::{float_tensor, long_tensor};
