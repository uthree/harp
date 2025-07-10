use crate::node::{Capture, Node, NodeData, Wildcard};
use std::collections::HashMap;
use std::sync::Arc;

/// A map to store captured nodes, mapping capture names to the matched nodes.
pub type Captures = HashMap<String, Node>;

/// The body of a rewriter, which can be a static pattern or a dynamic function.
pub enum RewriterBody {
    /// A static pattern to replace with.
    Pattern(Node),
    /// A function that takes captures and returns a new node.
    Func(Box<dyn Fn(&Captures) -> Option<Node>>),
}

/// A rule for rewriting a `Node` graph.
pub struct RewriteRule {
    /// The pattern to search for, represented as a `Node`.
    pub searcher: Node,
    /// The body of the rewriter.
    pub rewriter: RewriterBody,
}

impl RewriteRule {
    /// Creates a new rewrite rule with a static pattern.
    pub fn new(searcher: Node, rewriter: Node) -> Self {
        Self {
            searcher,
            rewriter: RewriterBody::Pattern(rewriter),
        }
    }

    /// Creates a new rewrite rule with a dynamic function.
    pub fn new_fn(searcher: Node, rewriter: impl Fn(&Captures) -> Option<Node> + 'static) -> Self {
        Self {
            searcher,
            rewriter: RewriterBody::Func(Box::new(rewriter)),
        }
    }
}

/// Creates a `RewriteRule` more concisely.
///
/// # Example
///
/// ```
/// use harp::node::{self, Node};
/// use harp::pattern::RewriteRule;
/// use harp::rewrite_rule;
///
/// let rule = rewrite_rule!(let x = capture("x"); node::recip(node::recip(x.clone())) => x);
/// ```
#[macro_export]
macro_rules! rewrite_rule {
    ( $(let $var:ident = capture($name:literal));* ; $searcher:expr => $rewriter:expr ) => {
        {
            $(let $var = $crate::node::capture($name);)*
            $crate::pattern::RewriteRule::new($searcher, $rewriter)
        }
    };
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
    pub fn rewrite(&self, node: Node) -> Node {
        // First, rewrite the children recursively.
        let rewritten_src = node
            .src()
            .iter()
            .map(|child| self.rewrite(child.clone()))
            .collect::<Vec<_>>();

        let mut current_node = Node::from(Arc::new(NodeData {
            op: node.op().clone(),
            src: rewritten_src,
        }));

        // Then, try to apply rules to the current node.
        for rule in &self.rules {
            if let Some(rewritten) = self.apply_rule(&current_node, rule) {
                current_node = rewritten;
            }
        }

        current_node
    }

    /// Tries to apply a single rule to a node.
    fn apply_rule(&self, node: &Node, rule: &RewriteRule) -> Option<Node> {
        let mut captures = Captures::new();
        if self.match_pattern(&rule.searcher, node, &mut captures) {
            match &rule.rewriter {
                RewriterBody::Pattern(pattern) => self.build_from_pattern(pattern, &captures),
                RewriterBody::Func(func) => func(&captures),
            }
        } else {
            None
        }
    }

    /// Matches a pattern node against a graph node, populating captures.
    fn match_pattern(&self, pattern: &Node, node: &Node, captures: &mut Captures) -> bool {
        // Handle Wildcard
        if pattern.op().as_any().is::<Wildcard>() {
            return true;
        }

        // Handle Capture
        if let Some(capture) = pattern.op().as_any().downcast_ref::<Capture>() {
            captures
                .entry(capture.0.clone())
                .or_insert_with(|| node.clone());
            return true;
        }

        // General operator matching
        if pattern.op().name() == node.op().name() && pattern.src().len() == node.src().len() {
            pattern
                .src()
                .iter()
                .zip(node.src().iter())
                .all(|(p, n)| self.match_pattern(p, n, captures))
        } else {
            false
        }
    }

    /// Builds a new node from a pattern and captured nodes.
    fn build_from_pattern(&self, pattern: &Node, captures: &Captures) -> Option<Node> {
        // If the pattern is a capture, retrieve the captured node.
        if let Some(capture) = pattern.op().as_any().downcast_ref::<Capture>() {
            return captures.get(&capture.0).cloned();
        }

        // Otherwise, build a new node, recursively building its sources.
        let new_src = pattern
            .src()
            .iter()
            .map(|p| self.build_from_pattern(p, captures))
            .collect::<Option<Vec<_>>>()?;

        Some(Node::from(Arc::new(NodeData {
            op: pattern.op().clone(),
            src: new_src,
        })))
    }
}
