//! # Harp API
//!
//! This crate provides the public, user-facing API for `harp`.
//! It is the primary entry point for users of the library.

pub mod tensor;

/// A prelude for conveniently importing the most common items.
pub mod prelude {
    pub use crate::tensor::Tensor;
    pub use harp_ir::prelude::*;
}