//! Graph module for computation graph representation
//!
//! This module provides the core data structures for representing
//! computation graphs in Harp.
//!
//! # Overview
//!
//! The computation graph is a DAG (Directed Acyclic Graph) where:
//! - Nodes (`GraphNode`) represent operations or data sources
//! - Edges represent data dependencies (via the `src` field)
//!
//! # Key Types
//!
//! - `GraphNode`: A computation node (shared via `Rc`)
//! - `GraphOp`: The operation type (View or MapReduce)
//! - `ReduceOp`: Reduction operations (Sum, Max, Min, Prod)
//! - `View`: Memory layout representation
//! - `Expr`: Symbolic shape expressions
//!
//! # Example
//!
//! ```rust
//! use eclat::graph::{input, Expr, DType};
//!
//! // Create input tensors
//! let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
//! let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
//!
//! // Build computation graph
//! let c = &a + &b;
//! let d = c.sum(1);
//! ```

// Shape submodule (moved from src/shape/)
pub mod shape;

// Graph node definitions
mod builder;
mod node;
mod ops;
mod traversal;

#[cfg(test)]
mod tests;

// ============================================================================
// Re-exports
// ============================================================================

// Shape types
pub use shape::{Expr, PadValue, View};

// Node types
pub use node::{GraphInner, GraphNode, GraphOp, ReduceOp};

// Builder functions
pub use builder::{
    GraphNodeBuilder, constant, dynamic_input, input, named_input, ones, scalar, zeros,
};

// Traversal utilities
pub use traversal::{
    GraphTransform, NodeReplacer, TraversalOrder, collect_inputs, collect_nodes, count_nodes,
    find_common_subexpressions, graph_to_string, has_cycle, topological_sort,
};

// Re-export DType for convenience
pub use crate::ast::DType;
