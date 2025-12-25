//! Core types and utilities
//!
//! This module contains fundamental types used throughout the crate.
//! These types are independent of specific computation backends.

mod dtype;

pub use dtype::DType;

// Note: Expr and View remain in graph::shape for now due to dependencies on ast and graph::ops.
// They will be moved here when the graph module is deleted in Phase 8.
// For now, use crate::graph::shape::{Expr, View} for these types.
