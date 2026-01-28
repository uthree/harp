//! Graph-level suggesters for optimization
//!
//! Each suggester detects specific patterns in the computation graph
//! and proposes optimized transformations.

mod composite;
mod fusion_adapter;
mod matmul;

pub use composite::CompositeSuggester;
pub use fusion_adapter::FusionSuggester;
pub use matmul::MatMulDetectorSuggester;
