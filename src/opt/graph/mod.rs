use crate::graph::Graph;

pub trait GraphOptimizer {
    fn new() -> Self;
    fn optimize(&mut self, graph: &Graph) -> Graph;
}
