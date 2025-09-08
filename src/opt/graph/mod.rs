use crate::graph::Graph;

/// A trait for graph optimizers.
pub trait GraphOptimizer {
    fn optimize(&self, graph: &Graph) -> Graph;
}

/// An optimizer that combines multiple graph optimizers.
pub struct CombinedGraphOptimizer {
    optimizers: Vec<Box<dyn GraphOptimizer>>,
}

impl CombinedGraphOptimizer {
    pub fn new(optimizers: Vec<Box<dyn GraphOptimizer>>) -> Self {
        Self { optimizers }
    }
}

impl GraphOptimizer for CombinedGraphOptimizer {
    fn optimize(&self, graph: &Graph) -> Graph {
        self.optimizers
            .iter()
            .fold(graph.clone(), |g, optimizer| optimizer.optimize(&g))
    }
}
