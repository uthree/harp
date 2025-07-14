//! A builder for creating `PatternGraph` instances fluently.

use crate::pattern::{PatternEdge, PatternGraph, PatternNode};

/// A builder for `PatternGraph`.
pub struct PatternBuilder<T, E> {
    node: PatternNode<T>,
    children: Vec<(PatternEdge<E>, PatternBuilder<T, E>)>,
}

impl<T, E> PatternBuilder<T, E>
where
    T: Clone,
    E: Clone,
{
    /// Creates a new builder for a node with specific data.
    pub fn new(data: T) -> Self {
        Self {
            node: PatternNode::Node { data },
            children: Vec::new(),
        }
    }

    /// Creates a new builder for a wildcard node.
    pub fn wildcard() -> Self {
        Self {
            node: PatternNode::Wildcard,
            children: Vec::new(),
        }
    }

    /// Creates a new builder for a capture node.
    pub fn capture(name: &str) -> Self {
        Self {
            node: PatternNode::Capture {
                name: name.to_string(),
            },
            children: Vec::new(),
        }
    }

    /// Adds a child to the pattern node.
    pub fn child(mut self, edge: E, child_builder: PatternBuilder<T, E>) -> Self {
        self.children
            .push((PatternEdge::Edge { data: edge }, child_builder));
        self
    }

    /// Adds a child with a wildcard edge.
    pub fn wildcard_child(mut self, child_builder: PatternBuilder<T, E>) -> Self {
        self.children.push((PatternEdge::Wildcard, child_builder));
        self
    }

    /// Consumes the builder and creates a `PatternGraph`.
    pub fn build(self) -> PatternGraph<T, E> {
        let mut pattern_graph = PatternGraph::new(self.node);
        let root = pattern_graph.root;

        for (edge, child_builder) in self.children {
            Self::build_recursive(&mut pattern_graph, root, edge, child_builder);
        }

        pattern_graph
    }

    /// Helper function to recursively build the pattern graph.
    fn build_recursive(
        graph: &mut PatternGraph<T, E>,
        parent_id: crate::NodeId,
        edge: PatternEdge<E>,
        builder: PatternBuilder<T, E>,
    ) {
        let child_id = graph.graph.add_node(builder.node);
        graph.graph.add_edge(parent_id, child_id, edge);

        for (edge, child_builder) in builder.children {
            Self::build_recursive(graph, child_id, edge, child_builder);
        }
    }
}