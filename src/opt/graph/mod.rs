pub mod fusion;

use crate::graph::Graph;

pub trait GraphOptimizer {
    fn optimize(&mut self, _graph: &mut Graph) {}
}

pub use fusion::{GraphFusionOptimizer, OptimizationSnapshot};

/// VIZ環境変数が有効かチェック
pub fn is_viz_enabled() -> bool {
    std::env::var("VIZ").map(|v| v == "1").unwrap_or(false)
}
