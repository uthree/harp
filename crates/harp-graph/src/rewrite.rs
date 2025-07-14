//! Implements the graph searching and rewriting logic.

use crate::pattern::{Match, PatternGraph, PatternNode};
use crate::{Graph, NodeId};
use std::collections::HashMap;

impl<T> Graph<T>
where
    T: PartialEq + Clone,
{
    /// Finds all occurrences of a pattern in the graph.
    pub fn find_matches(&self, pattern: &PatternGraph<T>) -> Vec<Match> {
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
        pattern_graph: &Graph<PatternNode<T>>,
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

        for (graph_child_id, pattern_child_id) in
            graph_node.children.iter().zip(pattern_node.children.iter())
        {
            if !self.match_node(*graph_child_id, *pattern_child_id, pattern_graph, captures) {
                return false;
            }
        }

        true
    }

    /// Rewrites the graph by finding all matches for a pattern and applying a rewriter function.
    pub fn rewrite<F>(&mut self, pattern: &PatternGraph<T>, mut rewriter: F)
    where
        F: FnMut(&mut Self, &Match) -> NodeId,
    {
        let matches = self.find_matches(pattern);
        for m in matches {
            let new_root_id = rewriter(self, &m);
            let old_root_id = m.root;

            // Do not replace if the rewriter returns the same node.
            if new_root_id == old_root_id {
                continue;
            }

            // Find all nodes that have the old root as a child and replace it.
            for i in 0..self.nodes.len() {
                let node_id = NodeId(i);
                if let Some(node) = self.get_mut(node_id) {
                    for child_id in &mut node.children {
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
    use super::*;
    use crate::pattern::*;

    // Helper to build a simple graph for testing
    // A -> B -> C
    // |
    // D
    fn build_test_graph() -> Graph<String> {
        let mut graph = Graph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        let d = graph.add_node("D".to_string());
        graph.add_edge(a, b);
        graph.add_edge(b, c);
        graph.add_edge(a, d);
        graph
    }

    #[test]
    fn test_find_simple_match() {
        let graph = build_test_graph();
        // Pattern: A -> B
        let mut pattern = PatternGraph::new(PatternNode::Node { data: "A".to_string() });
        pattern.add_child(pattern.root, PatternNode::Node { data: "B".to_string() });

        let matches = graph.find_matches(&pattern);
        assert_eq!(matches.len(), 0); // This should be 0 because A has two children
    }

    #[test]
    fn test_find_exact_match() {
        let mut graph = Graph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        graph.add_edge(a,b);

        let mut pattern = PatternGraph::new(PatternNode::Node {data: "A".to_string()});
        pattern.add_child(pattern.root, PatternNode::Node {data: "B".to_string()});
        let matches = graph.find_matches(&pattern);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].root, a);
    }


    #[test]
    fn test_find_with_wildcard() {
        let graph = build_test_graph();
        // Pattern: B -> *
        let mut pattern_b = PatternGraph::new(PatternNode::Node { data: "B".to_string() });
        pattern_b.add_child(pattern_b.root, PatternNode::Wildcard);

        let matches = graph.find_matches(&pattern_b);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].root, NodeId(1));
        assert_eq!(graph.get(matches[0].root).unwrap().children[0], NodeId(2));
    }

    #[test]
    fn test_find_with_capture() {
        let graph = build_test_graph();
        // Pattern: B -> capture("captured_node")
        let mut pattern = PatternGraph::new(PatternNode::Node { data: "B".to_string() });
        pattern.add_child(
            pattern.root,
            PatternNode::Capture {
                name: "captured_node".to_string(),
            },
        );

        let matches = graph.find_matches(&pattern);
        assert_eq!(matches.len(), 1);
        let m = &matches[0];
        assert_eq!(m.root, NodeId(1));
        assert_eq!(
            m.captures.get("captured_node").unwrap(),
            &NodeId(2) // C should be captured
        );
    }

    #[test]
    fn test_rewrite_simple() {
        let mut graph = build_test_graph();
        // Pattern: B -> C
        let mut pattern = PatternGraph::new(PatternNode::Node { data: "B".to_string() });
        pattern.add_child(pattern.root, PatternNode::Node { data: "C".to_string() });

        // Rewrite B -> C with a new node E
        graph.rewrite(&pattern, |g, _match| {
            g.add_node("E".to_string())
        });

        // A should now point to E and D.
        let node_a = graph.get(NodeId(0)).unwrap();
        // The rewrite logic replaces references to B (the root of the match)
        // with E. So A's child B becomes E.
        assert!(node_a.children.contains(&NodeId(4))); // E
        assert!(!node_a.children.contains(&NodeId(1))); // B
        assert!(node_a.children.contains(&NodeId(3))); // D
    }

    #[test]
    fn test_rewrite_with_capture() {
        // X -> Y
        let mut graph = Graph::new();
        let x = graph.add_node("X".to_string());
        let y = graph.add_node("Y".to_string());
        graph.add_edge(x, y);

        // Pattern: X -> capture("the_child")
        let mut pattern = PatternGraph::new(PatternNode::Node { data: "X".to_string() });
        pattern.add_child(pattern.root, PatternNode::Capture { name: "the_child".to_string() });

        graph.rewrite(&pattern, |g, m| {
            let captured_id = *m.captures.get("the_child").unwrap();
            let captured_data = g.get(captured_id).unwrap().data.clone();
            // Create a new node with the data of the captured node + " (rewritten)"
            let new_data = format!("{} (rewritten)", captured_data);
            g.add_node(new_data)
        });

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.get(NodeId(2)).unwrap().data, "Y (rewritten)");
    }
}
