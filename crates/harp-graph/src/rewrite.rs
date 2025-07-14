//! Implements the graph searching and rewriting logic.

use crate::pattern::{Match, PatternEdge, PatternGraph, PatternNode};
use crate::{Graph, NodeId};
use std::collections::HashMap;

impl<T, E> Graph<T, E>
where
    T: PartialEq + Clone,
    E: PartialEq + Clone,
{
    /// Finds all occurrences of a pattern in the graph.
    pub fn find_matches(&self, pattern: &PatternGraph<T, E>) -> Vec<Match> {
        let mut matches = Vec::new();
        for i in 0..self.nodes.len() {
            let root_id = NodeId(i);
            let mut captures = HashMap::new();
            if self.match_node(root_id, pattern.root, &pattern.graph, &mut captures) {
                matches.push(Match {
                    root: root_id,
                    captures,
                });
            }
        }
        matches
    }

    /// A helper function to recursively match a pattern node against a graph node.
    fn match_node(
        &self,
        graph_node_id: NodeId,
        pattern_node_id: NodeId,
        pattern_graph: &Graph<PatternNode<T>, PatternEdge<E>>,
        captures: &mut HashMap<String, NodeId>,
    ) -> bool {
        let graph_node = self.get(graph_node_id).unwrap();
        let pattern_node = pattern_graph.get(pattern_node_id).unwrap();

        // Match the node data itself
        let data_matches = match &pattern_node.data {
            PatternNode::Node { data } => *data == graph_node.data,
            PatternNode::Wildcard => true,
            PatternNode::Capture { name } => {
                captures.insert(name.clone(), graph_node_id);
                true
            }
        };

        if !data_matches {
            // Backtrack captures if data doesn't match
            if let PatternNode::Capture { name } = &pattern_node.data {
                captures.remove(name);
            }
            return false;
        }

        // Match children
        if graph_node.children.len() != pattern_node.children.len() {
            // Backtrack captures if children count doesn't match
            if let PatternNode::Capture { name } = &pattern_node.data {
                captures.remove(name);
            }
            return false;
        }

        for ((graph_edge_data, graph_child_id), (pattern_edge_data, pattern_child_id)) in
            graph_node.children.iter().zip(pattern_node.children.iter())
        {
            // Match edge data
            let edge_matches = match pattern_edge_data {
                PatternEdge::Edge { data } => data == graph_edge_data,
                PatternEdge::Wildcard => true,
            };

            if !edge_matches {
                return false;
            }

            // Recursively match child node
            if !self.match_node(*graph_child_id, *pattern_child_id, pattern_graph, captures) {
                return false;
            }
        }

        true
    }

    /// Rewrites the graph by finding all matches for a pattern and applying a rewriter function.
    pub fn rewrite<F>(&mut self, pattern: &PatternGraph<T, E>, mut rewriter: F)
    where
        F: FnMut(&mut Self, &Match) -> NodeId,
    {
        let matches = self.find_matches(pattern);
        for m in matches {
            let new_root_id = rewriter(self, &m);
            let old_root_id = m.root;

            if new_root_id == old_root_id {
                continue;
            }

            for i in 0..self.nodes.len() {
                let node_id = NodeId(i);
                if let Some(node) = self.get_mut(node_id) {
                    for (_, child_id) in &mut node.children {
                        if *child_id == old_root_id {
                            *child_id = new_root_id;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::PatternBuilder;
    use crate::*;

    // A(op: Sub) -> [ (0, B), (1, C) ]  (B - C)
    fn build_test_graph() -> Graph<String, usize> {
        let mut graph = Graph::new();
        let a = graph.add_node("Sub".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, 0); // B is the 0th operand
        graph.add_edge(a, c, 1); // C is the 1st operand
        graph
    }

    #[test]
    fn test_find_match_with_edge_data() {
        let graph = build_test_graph();
        let pattern = PatternBuilder::new("Sub".to_string())
            .child(0, PatternBuilder::new("B".to_string()))
            .child(1, PatternBuilder::new("C".to_string()))
            .build();

        let matches = graph.find_matches(&pattern);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].root, NodeId(0));
    }

    #[test]
    fn test_find_match_with_wrong_edge_data() {
        let graph = build_test_graph();
        let pattern = PatternBuilder::new("Sub".to_string())
            .child(1, PatternBuilder::new("B".to_string()))
            .child(0, PatternBuilder::new("C".to_string()))
            .build();

        let matches = graph.find_matches(&pattern);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_find_with_capture_and_wildcard() {
        let graph = build_test_graph();
        let pattern = PatternBuilder::new("Sub".to_string())
            .child(0, PatternBuilder::capture("b"))
            .wildcard_child(PatternBuilder::wildcard())
            .build();

        let matches = graph.find_matches(&pattern);
        assert_eq!(matches.len(), 1);
        let m = &matches[0];
        assert_eq!(m.root, NodeId(0));
        assert_eq!(m.captures.get("b").unwrap(), &NodeId(1));
    }

    #[test]
    fn test_rewrite_with_edge_data() {
        let mut graph = build_test_graph();
        let pattern = PatternBuilder::new("Sub".to_string())
            .child(0, PatternBuilder::new("B".to_string()))
            .child(1, PatternBuilder::new("C".to_string()))
            .build();

        graph.rewrite(&pattern, |g, _match| g.add_node("Rewritten".to_string()));

        assert_eq!(graph.len(), 4);
        assert_eq!(graph.get(NodeId(3)).unwrap().data, "Rewritten");
    }
}
