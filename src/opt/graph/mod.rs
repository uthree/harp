use crate::graph::Graph;

pub trait GraphOptimizer {
    fn optimize(&mut self, graph: &Graph) -> Graph;
}
