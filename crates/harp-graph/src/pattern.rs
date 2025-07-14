//! Defines the structures for pattern matching.

use crate::{Graph, NodeId};
use std::collections::HashMap;

/// Represents a node in a pattern graph.
#[derive(Debug, Clone)]
pub enum PatternNode<T> {
    /// Matches a node with specific data.
    Node { data: T },
    /// A wildcard that matches any node.
    Wildcard,
    /// A capture node that matches any node and captures its ID.
    Capture { name: String },
}

/// Represents an edge in a pattern graph.
#[derive(Debug, Clone)]
pub enum PatternEdge<E> {
    /// Matches an edge with specific data.
    Edge { data: E },
    /// A wildcard that matches any edge.
    Wildcard,
}

/// A graph structure that represents a pattern to be matched.
pub struct PatternGraph<T, E> {
    pub(crate) graph: Graph<PatternNode<T>, PatternEdge<E>>,
    pub(crate) root: NodeId,
}

impl<T, E> PatternGraph<T, E> {
    /// Creates a new pattern with a root node.
    pub fn new(root_node: PatternNode<T>) -> Self {
        let mut graph = Graph::new();
        let root = graph.add_node(root_node);
        Self { graph, root }
    }

    /// Adds a child to the pattern.
    pub fn add_child(
        &mut self,
        parent: NodeId,
        edge: PatternEdge<E>,
        child_node: PatternNode<T>,
    ) -> NodeId {
        let child_id = self.graph.add_node(child_node);
        self.graph.add_edge(parent, child_id, edge);
        child_id
    }
}

/// Represents a successful match of a pattern in a graph.
#[derive(Debug, Clone)]
pub struct Match {
    /// The root node of the matched subgraph.
    pub root: NodeId,
    /// A map from capture names to the captured node IDs.
    pub captures: HashMap<String, NodeId>,
}
