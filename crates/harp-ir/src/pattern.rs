use crate::node::{Node, NodeData};
use crate::op::{Capture, Wildcard};
use dyn_clone::clone_box;
use std::collections::HashMap;
use std::ops::Add;
use std::rc::Rc;

/// A map to store captured nodes, mapping capture names to the matched nodes.
pub type Captures = HashMap<String, Node>;

/// A function that takes captures and returns a new node.
pub type RewriterFn = Box<dyn Fn(&Node, &Captures) -> Option<Node>>;

/// The body of a rewriter, which can be a static pattern or a dynamic function.
pub enum RewriterBody {
    /// A static pattern to replace with.
    Pattern(Node),
    /// A function that takes captures and returns a new node.
    Func(RewriterFn),
}

impl PartialEq for RewriterBody {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RewriterBody::Pattern(p1), RewriterBody::Pattern(p2)) => p1 == p2,
            // Functions are considered unique and cannot be compared for equality.
            _ => false,
        }
    }
}
impl Eq for RewriterBody {}

/// A rule for rewriting a `Node` graph.
///
/// A `RewriteRule` consists of a `searcher` pattern and a `rewriter` body.
/// When the `searcher` pattern matches a part of the graph, the `rewriter`
/// is used to generate the replacement node.
#[derive(PartialEq, Eq)]
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
/// use harp_ir::node::{self, Node};
/// use harp_ir::pattern::RewriteRule;
/// use harp_ir::rewrite_rule;
///
/// let rule = rewrite_rule!(let x = capture("x"); node::recip(node::recip(x.clone())) => x);
/// ```
#[macro_export]
#[deprecated(
    since = "0.1.0",
    note = "Please use the `rewriter!` macro instead. It provides a more unified and expressive way to define rules."
)]
macro_rules! rewrite_rule {
    ( $(let $var:ident = capture($name:literal));* ; $searcher:expr => $rewriter:expr ) => {
        {
            $(let $var = $crate::node::capture($name);)*
            $crate::pattern::RewriteRule::new($searcher, $rewriter)
        }
    };
}

/// Applies a set of rewrite rules to a `Node` graph in a hierarchical manner.
///
/// A `Rewriter` has a name for debugging, a list of its own `RewriteRule`s,
/// and a list of sub-rewriters. This creates a hierarchical structure for
/// organizing rules. When applying rules, the `Rewriter` traverses this
/// hierarchy and applies all unique rules.
pub struct Rewriter {
    pub name: String,
    pub rules: Vec<RewriteRule>,
    pub sub_rewriters: Vec<Rewriter>,
}

impl Rewriter {
    /// Creates a new rewriter with a given name and a set of rules.
    pub fn new(name: impl Into<String>, rules: Vec<RewriteRule>) -> Self {
        Self {
            name: name.into(),
            rules,
            sub_rewriters: Vec::new(),
        }
    }

    /// Recursively collects all unique rewrite rules from the hierarchy.
    pub fn get_all_rules<'a>(&'a self, all_rules: &mut Vec<&'a RewriteRule>) {
        // Add rules from the current rewriter, avoiding duplicates.
        for rule in &self.rules {
            if !all_rules.contains(&rule) {
                all_rules.push(rule);
            }
        }

        // Recursively add rules from sub-rewriters.
        for sub_rewriter in &self.sub_rewriters {
            sub_rewriter.get_all_rules(all_rules);
        }
    }

    /// Applies the rules to a node and its descendants, returning a rewritten node.
    ///
    /// The rewriting process is bottom-up: children are rewritten first, and then
    /// the rules are applied to the current node until a fixed point is reached.
    pub fn rewrite(&self, node: Node) -> Node {
        // 1. Rewrite children first (bottom-up)
        let rewritten_src = node
            .src()
            .iter()
            .map(|child| self.rewrite(child.clone()))
            .collect::<Vec<_>>();

        let mut current_node = if !rewritten_src.iter().zip(node.src()).all(|(a, b)| a == b) {
            Node::from(Rc::new(NodeData {
                op: clone_box(node.op()),
                src: rewritten_src,
            }))
        } else {
            node
        };

        // 2. Collect all unique rules from the hierarchy.
        let mut all_rules = Vec::new();
        self.get_all_rules(&mut all_rules);

        // 3. Apply rules to the current node until a fixed point is reached.
        loop {
            let mut changed = false;
            for rule in &all_rules {
                if let Some(rewritten) = self.apply_rule(&current_node, rule) {
                    log::debug!(
                        "[Rewrite by {}] {current_node:?} -> {rewritten:?}",
                        self.name
                    );
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
        if Self::match_pattern(&rule.searcher, node, &mut captures) {
            match &rule.rewriter {
                RewriterBody::Pattern(pattern) => Self::build_from_pattern(pattern, &captures),
                RewriterBody::Func(func) => func(node, &captures),
            }
        } else {
            None
        }
    }

    /// Matches a pattern node against a graph node, populating captures.
    fn match_pattern(pattern: &Node, node: &Node, captures: &mut Captures) -> bool {
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
                .all(|(p, n)| Self::match_pattern(p, n, captures))
        } else {
            false
        }
    }

    /// Builds a new node from a pattern and captured nodes.
    fn build_from_pattern(pattern: &Node, captures: &Captures) -> Option<Node> {
        // If the pattern is a capture, retrieve the captured node.
        if let Some(capture) = pattern.op().as_any().downcast_ref::<Capture>() {
            return captures.get(&capture.0).cloned();
        }

        // Otherwise, build a new node, recursively building its sources.
        let new_src = pattern
            .src()
            .iter()
            .map(|p| Self::build_from_pattern(p, captures))
            .collect::<Option<Vec<_>>>()?;

        Some(Node::from(Rc::new(NodeData {
            op: clone_box(pattern.op()),
            src: new_src,
        })))
    }
}

impl Add for Rewriter {
    type Output = Self;

    /// Combines two rewriters into a new hierarchical rewriter.
    ///
    /// The resulting rewriter will have a new name and will contain the
    /// two original rewriters as sub-rewriters.
    fn add(self, rhs: Self) -> Self::Output {
        let name = format!("fused({}, {})", self.name, rhs.name);
        Rewriter {
            name,
            rules: Vec::new(),
            sub_rewriters: vec![self, rhs],
        }
    }
}
