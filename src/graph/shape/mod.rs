//! Shape types for tensor operations
//!
//! This module contains the core shape types used for tensor computations.

pub mod expr;
pub mod view;

#[cfg(test)]
mod tests;

pub use expr::Expr;
pub use view::{PadValue, View, ViewBounds};
