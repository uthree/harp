//! Graph-level suggesters for optimization
//!
//! Each suggester detects specific patterns in the computation graph
//! and proposes optimized transformations.

mod composite;
mod elementwise_reduce_fusion;
mod matmul;
mod view_fusion;

pub use composite::CompositeSuggester;
pub use elementwise_reduce_fusion::ElementwiseReduceFusionSuggester;
pub use matmul::MatMulDetectorSuggester;
pub use view_fusion::ViewFusionSuggester;
