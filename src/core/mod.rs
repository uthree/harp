//! Core types and utilities
//!
//! This module contains fundamental types used throughout the crate.
//! These types are independent of specific computation backends.

mod dtype;
pub mod shape;

pub use dtype::DType;
pub use shape::{Expr, View};
