use crate::graph::Graph;

pub trait GraphOptimizer {
    fn optimize(&mut self, _graph: &mut Graph) {}
}
