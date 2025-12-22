//! Harp DSL - Domain Specific Language for Harp computation graphs
//!
//! This crate provides a DSL for defining computation graphs that can be
//! compiled to optimized kernels for various platforms.
//!
//! # Example
//!
//! ```harp
//! graph<L, M, N> matmul(a: f32[L, M], b: f32[M, N]) -> (c: f32[L, N]) {
//!     let a_exp = a.unsqueeze(2)
//!     let b_exp = b.unsqueeze(0)
//!     let a_bc = a_exp.repeat(2, N)
//!     let b_bc = b_exp.repeat(0, L)
//!     let prod = a_bc * b_bc
//!     c = prod.sum(1)
//! }
//! ```

pub mod compiler;
pub mod decompiler;
pub mod error;
pub mod parser;

use error::DslError;
use harp_core::graph::Graph;

/// Parse DSL source code into a DSL AST
pub fn parse(source: &str) -> Result<parser::ast::DslModule, DslError> {
    parser::parse(source)
}

/// Compile DSL source code to a Harp Graph
pub fn compile(source: &str) -> Result<Graph, DslError> {
    let module = parse(source)?;
    compiler::compile(&module)
}

/// Decompile a Harp Graph to DSL source code
/// エントリーポイントとなるグラフは常に"main"という名前で出力される
pub fn decompile(graph: &Graph) -> String {
    decompiler::decompile(graph)
}
