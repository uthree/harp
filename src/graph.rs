use petgraph::prelude::{Directed, StableGraph};

#[derive(Debug, Clone)]
pub enum Operator {
    Input,
}

#[derive(Debug, Clone)]
pub struct Node {}

type StorageGraph = StableGraph<Box<Node>, usize, Directed, usize>;

#[derive(Debug, Clone, Default)]

pub struct Graph {
    body: StorageGraph,
}

// Computation graph
impl Graph {
    fn new() -> Graph {
        Self::default()
    }
}
