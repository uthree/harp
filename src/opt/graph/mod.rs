//! Graph-level optimization framework
//!
//! This module provides optimization passes that operate on the computation graph
//! before lowering to AST. Graph-level optimizations can detect high-level patterns
//! (like matrix multiplication) and replace them with optimized operations.

pub mod estimator;
pub mod optimizer;
pub mod suggesters;

use crate::graph::GraphNode;

/// Result of a graph transformation suggestion
#[derive(Clone, Debug)]
pub struct GraphSuggestResult {
    /// The transformed graph roots
    pub roots: Vec<GraphNode>,
    /// Name of the suggester that produced this result
    pub suggester_name: String,
    /// Description of the transformation
    pub description: String,
}

impl GraphSuggestResult {
    /// Create a new GraphSuggestResult
    pub fn new(roots: Vec<GraphNode>, suggester_name: impl Into<String>) -> Self {
        Self {
            roots,
            suggester_name: suggester_name.into(),
            description: String::new(),
        }
    }

    /// Create a new GraphSuggestResult with description
    pub fn with_description(
        roots: Vec<GraphNode>,
        suggester_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            roots,
            suggester_name: suggester_name.into(),
            description: description.into(),
        }
    }
}

/// Trait for suggesting graph transformations
///
/// A GraphSuggester analyzes the computation graph and proposes
/// transformations that may improve performance.
pub trait GraphSuggester {
    /// Get the name of this suggester
    fn name(&self) -> &str;

    /// Suggest possible transformations for the given graph roots
    ///
    /// Returns a list of possible transformed graphs, each potentially
    /// with different trade-offs.
    fn suggest(&self, roots: &[GraphNode]) -> Vec<GraphSuggestResult>;
}

/// Trait for estimating the cost of a computation graph
///
/// Used by beam search to evaluate and compare different graph transformations.
pub trait GraphCostEstimator {
    /// Estimate the execution cost of the graph
    ///
    /// Returns a cost value in log scale (lower is better).
    fn estimate(&self, roots: &[GraphNode]) -> f32;
}

/// Trait for optimizing computation graphs
pub trait GraphOptimizer {
    /// Optimize the graph and return the optimized roots
    fn optimize(&mut self, roots: Vec<GraphNode>) -> Vec<GraphNode>;
}

// Re-exports
pub use estimator::SimpleGraphCostEstimator;
pub use optimizer::GraphBeamSearchOptimizer;
pub use suggesters::{CompositeSuggester, FusionSuggester, MatMulDetectorSuggester};
