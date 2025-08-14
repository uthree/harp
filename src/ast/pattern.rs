use std::rc::Rc;

use crate::ast::AstNode;

pub struct RewriteRule {
    pattern: AstNode,
    rewriter: Box<dyn Fn(&[AstNode]) -> AstNode>,
    /// A closure that takes captured nodes and returns true if the rewrite should be applied.
    condition: Box<dyn Fn(&[AstNode]) -> bool>,
}

impl RewriteRule {
    pub fn new(
        pattern: AstNode,
        rewriter: impl Fn(&[AstNode]) -> AstNode + 'static,
        condition: impl Fn(&[AstNode]) -> bool + 'static,
    ) -> Rc<Self> {
        Rc::new(RewriteRule {
            pattern: pattern,
            rewriter: Box::new(rewriter),
            condition: Box::new(condition),
        })
    }
}

/// A macro to create a `RewriteRule`.
///
/// # Example
///
/// The following example is ignored because `astpat!` is not exported.
/// ```ignore
/// use crate::ast::AstNode;
/// use crate::ast::pattern::RewriteRule;
///
/// // without condition
/// let rule = astpat!(|a| a + 1 => a);
///
/// // with condition
/// let rule = astpat!(|a, b| a + b, if a == b => b + a);
/// ```
macro_rules! astpat {
    (| $($capture: pat_param),* | $pattern: expr, if $condition: expr => $rewriter: expr) => {
        {
            let mut counter = 0..;
            $(
                let $capture = AstNode::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = |captured_nodes: &[AstNode]| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $rewriter
            };
            let condition = |captured_nodes: &[AstNode]| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $condition
            };
            RewriteRule::new(pattern, rewriter, condition)
        }
    };
    (| $($capture: pat_param),* | $pattern: expr => $rewriter: expr ) => {
        {
            let mut counter = 0..;
            $(
                let $capture = AstNode::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = |captured_nodes: &[AstNode]| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $rewriter
            };
            RewriteRule::new(pattern, rewriter, |_| true)
        }
    };
}

pub struct AstRewriter {
    name: String,
    rules: Vec<Rc<RewriteRule>>,
}
