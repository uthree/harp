//! # Harp: A Graph Computation and Rewriting Toolkit
//!
//! `harp` is a lightweight library for building and manipulating computation graphs.
//! It provides a simple and intuitive API for creating graph nodes, combining them
//! using overloaded operators, and rewriting them based on a set of powerful,
//! pattern-matching rules.
//!
//! ## Features
//!
//! - **Easy Graph Construction**: Build complex graphs effortlessly using familiar
//!   arithmetic operators (`+`, `*`, `-`, `/`).
//! - **Pattern Matching Rewriter**: Define structural and functional rewrite rules
//!   to simplify and optimize graphs.
//! - **Declarative Macros**: Use the `rewriter!` and `rewrite_rule!` macros to
//!   define rewrite rules in a clean, declarative style.
//! - **DOT Format Visualization**: Convert graphs to the DOT format to visualize
//!   their structure.
//! - **Extensible Core**: Define custom data types (`DType`) and operators (`Operator`)
//!   to suit your specific domain.
//! - **Flexible Logging**: Integrated with the `log` crate to provide detailed
//!   insight into the rewriting process.
//!
//! ## Quick Start
//!
//! Here's a simple example of building a graph and rewriting it.
//!
//! ```
//! use harp::node::{self, Node};
//! use harp::pattern::Rewriter;
//! use harp::rewriter;
//!
//! // 1. Define a rewriter with a rule: `x * 1.0 -> x`
//! let rewriter = rewriter!([
//!     (
//!         let x = capture("x")
//!         => x * Node::from(1.0f32)
//!         => |x| Some(x)
//!     )
//! ]);
//!
//! // 2. Build a graph: (a + b) * 1.0
//! let a = node::constant(10.0f32);
//! let b = node::constant(5.0f32);
//! let graph = (a + b) * 1.0;
//!
//! // 3. Apply the rewriter
//! let rewritten_graph = rewriter.rewrite(graph);
//!
//! // 4. Print the graph in DOT format
//! println!("{}", rewritten_graph.to_dot());
//! ```

pub mod macros;
pub mod node;
pub mod pattern;

