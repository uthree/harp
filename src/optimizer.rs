use crate::{
    graph::{EdgeMetadata, Graph},
    node::Node,
    operator,
    tensor::TensorData,
};
use petgraph::{Direction, algo::toposort, graph::NodeIndex, visit::EdgeRef};
use std::collections::HashMap;

/// Trait for graph optimization passes.
///
/// Any struct implementing this trait can be used to optimize a computation graph.
pub trait GraphOptimizer {
    /// Applies an optimization pass to the given graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - A mutable reference to the `Graph` to be optimized.
    ///
    /// # Returns
    ///
    /// `true` if the graph was modified, `false` otherwise.
    fn optimize(&mut self, graph: &mut Graph) -> bool;
}