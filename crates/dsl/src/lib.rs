//! Eclat DSL - Domain Specific Language for computation graphs
//!
//! This crate provides:
//! - AST types for the DSL (`ast`)
//! - Parser for DSL source code (`parser`)
//! - Error types (`errors`)
//! - Graph builder to convert DSL AST to GraphNode (`graph_builder`)

pub mod ast;
pub mod errors;
pub mod graph_builder;
pub mod parser;

pub use ast::*;
pub use errors::{DslError, DslResult};
pub use graph_builder::{BuiltGraph, GraphBuilder};
pub use parser::parse_program;
