//! # Harp IR
//!
//! This crate provides the core, shared data structures and logic for the
//! computation graph, serving as the "common language" between the frontend
//! API and the backend code generation.

pub mod dot;
pub mod dtype;
pub mod macros;
pub mod node;
pub mod op;
pub mod pattern;
pub mod prelude;
pub mod simplify;