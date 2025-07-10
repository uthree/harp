use crate::node::{Capture, Node, NodeData, Wildcard};
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;

/// A map to store captured nodes, mapping capture names to the matched nodes.
pub type Captures = HashMap<String, Node>;

/// The body of a rewriter, which can be a static pattern or a dynamic function.
pub enum RewriterBody {
    /// A static pattern to replace with.
    Pattern(Node),
    /// A function that takes captures and returns a new node.
    Func(Box<dyn Fn(&Node, &Captures) -> Option<Node>>),
}

/// A rule for rewriting a `Node` graph.
///
/// A `RewriteRule` consists of a `searcher` pattern and a `rewriter` body.
/// When the `searcher` pattern matches a part of the graph, the `rewriter`
/// is used to generate the replacement node.
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
    ///
    /// The function is passed a map of the captured nodes and should return
    /// a new `Node` if the rewrite is successful, or `None` otherwise.
    pub fn new_fn(
        searcher: Node,
        rewriter: impl Fn(&Node, &Captures) -> Option<Node> + 'static,
    ) -> Self {
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
///
/// A `Rewriter` contains a list of `RewriteRule`s and applies them to a graph
/// until a fixed point is reached.
pub struct Rewriter {
    rules: Vec<RewriteRule>,
}

impl Rewriter {
    /// Creates a new rewriter with a given set of rules.
    pub fn new(rules: Vec<RewriteRule>) -> Self {
        Self { rules }
    }

    /// Merges another rewriter's rules into this one.
    fn merge(&mut self, other: Rewriter) {
        self.rules.extend(other.rules);
    }

    /// Creates a new rewriter by merging two rewriters.
    pub fn fused(mut self, other: Rewriter) -> Self {
        self.merge(other);
        self
    }

    /// Applies the rules to a node and its descendants, returning a rewritten node.
    ///
    /// The rewriting process is bottom-up: children are rewritten first, and then
    /// the rules are applied to the current node until no more rules can be applied.
    pub fn rewrite(&self, node: Node) -> Node {
        // 1. Rewrite children first (bottom-up)
        let rewritten_src = node
            .src()
            .iter()
            .map(|child| self.rewrite(child.clone()))
            .collect::<Vec<_>>();

        let mut current_node = if !rewritten_src.iter().zip(node.src()).all(|(a, b)| a == b) {
            Node::from(Arc::new(NodeData {
                op: node.op().clone(),
                src: rewritten_src,
            }))
        } else {
            node
        };

        // 2. Apply rules to the current node until a fixed point is reached.
        loop {
            let mut changed = false;
            for rule in &self.rules {
                if let Some(rewritten) = self.apply_rule(&current_node, rule) {
                    log::debug!("[Rewrite] {current_node:?} -> {rewritten:?}");
                    current_node = rewritten;
                    changed = true;
                    // Restart the rule application process from the beginning for the new node
                    break;
                }
            }
            if !changed {
                break;
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
                RewriterBody::Func(func) => func(node, &captures),
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

impl Add for Rewriter {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.fused(rhs)
    }
}
