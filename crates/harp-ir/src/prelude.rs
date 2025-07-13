//! The `harp-ir` prelude.
//!
//! This module provides a convenient way to import the most commonly used
//! items from the `harp-ir` crate.
//!
//! # Example
//!
//! ```
//! use harp_ir::prelude::*;
//! ```

pub use crate::dot::ToDot;
pub use crate::dtype::DType;
pub use crate::node::{self, Node, capture, constant, variable};
pub use crate::op::*;
pub use crate::pattern::{RewriteRule, Rewriter};
pub use crate::simplify::simplify;
