use crate::node::{Capture, Node, Wildcard};
use std::collections::HashMap;
use std::sync::Arc;

/// A map to store captured nodes, mapping capture names to the matched nodes.
pub type Captures = HashMap<String, Arc<Node>>;

/// A rule for rewriting a `Node` graph.
pub struct RewriteRule {
    /// The pattern to search for, represented as a `Node`.
    pub searcher: Arc<Node>,
    /// The pattern to replace with, represented as a `Node`.
    pub rewriter: Arc<Node>,
}

impl RewriteRule {
    /// Creates a new rewrite rule.
    pub fn new(searcher: Arc<Node>, rewriter: Arc<Node>) -> Self {
        Self { searcher, rewriter }
    }
}

/// Applies a set of rewrite rules to a `Node` graph.
pub struct Rewriter {
    rules: Vec<RewriteRule>,
}

impl Rewriter {
    /// Creates a new rewriter with a given set of rules.
    pub fn new(rules: Vec<RewriteRule>) -> Self {
        Self { rules }
    }

    /// Applies the rules to a node and its descendants, returning a rewritten node.
    pub fn rewrite(&self, node: Arc<Node>) -> Arc<Node> {
        // First, rewrite the children recursively.
        let rewritten_src = node
            .src
            .iter()
            .map(|child| self.rewrite(child.clone()))
            .collect::<Vec<_>>();

        let mut current_node = Arc::new(Node {
            op: node.op.clone(),
            src: rewritten_src,
        });

        // Then, try to apply rules to the current node.
        for rule in &self.rules {
            if let Some(rewritten) = self.apply_rule(&current_node, rule) {
                current_node = rewritten;
            }
        }

        current_node
    }

    /// Tries to apply a single rule to a node.
    fn apply_rule(&self, node: &Arc<Node>, rule: &RewriteRule) -> Option<Arc<Node>> {
        let mut captures = Captures::new();
        if self.match_pattern(&rule.searcher, node, &mut captures) {
            self.build_from_pattern(&rule.rewriter, &captures)
        } else {
            None
        }
    }

    /// Matches a pattern node against a graph node, populating captures.
    fn match_pattern(
        &self,
        pattern: &Arc<Node>,
        node: &Arc<Node>,
        captures: &mut Captures,
    ) -> bool {
        // Handle Wildcard
        if pattern.op.as_any().is::<Wildcard>() {
            return true;
        }

        // Handle Capture
        if let Some(capture) = pattern.op.as_any().downcast_ref::<Capture>() {
            captures
                .entry(capture.0.clone())
                .or_insert_with(|| node.clone());
            return true;
        }

        // General operator matching
        if pattern.op.name() == node.op.name() && pattern.src.len() == node.src.len() {
            pattern
                .src
                .iter()
                .zip(node.src.iter())
                .all(|(p, n)| self.match_pattern(p, n, captures))
        } else {
            false
        }
    }

    /// Builds a new node from a pattern and captured nodes.
    fn build_from_pattern(&self, pattern: &Arc<Node>, captures: &Captures) -> Option<Arc<Node>> {
        // If the pattern is a capture, retrieve the captured node.
        if let Some(capture) = pattern.op.as_any().downcast_ref::<Capture>() {
            return captures.get(&capture.0).cloned();
        }

        // Otherwise, build a new node, recursively building its sources.
        let new_src = pattern
            .src
            .iter()
            .map(|p| self.build_from_pattern(p, captures))
            .collect::<Option<Vec<_>>>()?;

        Some(Arc::new(Node {
            op: pattern.op.clone(),
            src: new_src,
        }))
    }
}