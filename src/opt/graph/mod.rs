pub mod cost_estimator;
pub mod fusion;
pub mod optimizer;
pub mod suggester;

use crate::graph::Graph;

pub trait GraphOptimizer {
    fn optimize(&mut self, _graph: &mut Graph) {}
}

pub use cost_estimator::{estimate_graph_cost, estimate_node_cost};
pub use fusion::{GraphFusionOptimizer, OptimizationSnapshot};
pub use optimizer::{optimize_graph, optimize_graph_with_params, BeamSearchOptimizer};
pub use suggester::{
    CombinedSuggester, LoopPermutationSuggester, ParallelizationConfig, ParallelizationReason,
    ParallelizationSuggester, TileSize, TilingSuggester, VectorizationConfig,
    VectorizationSuggester,
};

/// VIZ環境変数が有効かチェック
pub fn is_viz_enabled() -> bool {
    std::env::var("VIZ").map(|v| v == "1").unwrap_or(false)
}
