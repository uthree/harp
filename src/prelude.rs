//! The `harp` prelude.
//!
//! This module contains the most commonly used types, traits, and functions
//! from the `harp` library, making them easily accessible with a single `use`
//! statement.
//!
//! # Example
//!
//! ```
//! use harp::prelude::*;
//!
//! // Now you can use Tensor, ClangBackend, DType, etc. directly.
//! ```

pub use crate::backends::{Backend, ClangBackend};
pub use crate::dot::ToDot;
pub use crate::dtype::DType;
pub use crate::shapetracker::ShapeTracker;
pub use crate::tensor::{Tensor, TensorOp};
