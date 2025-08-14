use std::rc::Rc;

use crate::ast::AstNode;

pub struct RewriteRule {
    pub pattern: AstNode,
    pub rewriter: Box<dyn Fn(&[AstNode]) -> AstNode>,
    /// A closure that takes captured nodes and returns true if the rewrite should be applied.
    pub condition: Box<dyn Fn(&[AstNode]) -> bool>,
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
/// ```
/// use harp::ast::AstNode;
/// use harp::ast::pattern::RewriteRule;
/// use harp::astpat;
///
/// // without condition
/// let rule = astpat!(|a| a + 1 => a);
///
/// // with condition
/// let rule = astpat!(|a, b| a + b, if a == b => b + a);
/// ```
#[macro_export]
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

#[test]
fn test_astpat_macro() {
    use crate::ast::AstNode;

    // Test case 1: Simple rule without condition
    let rule1 = astpat!(|a| a => a);
    let captured_nodes1 = [AstNode::from(1usize)];
    assert!((rule1.condition)(&captured_nodes1)); // Should always be true
    assert_eq!((rule1.rewriter)(&captured_nodes1), AstNode::from(1usize));

    // Test case 2: Rule with a condition
    let rule2 = astpat!(|a, b| a + b, if a == AstNode::from(1usize) => b + a);

    // Condition should be true
    let captured_nodes2_true = [AstNode::from(1usize), AstNode::from(2usize)];
    assert!((rule2.condition)(&captured_nodes2_true));
    assert_eq!(
        (rule2.rewriter)(&captured_nodes2_true),
        AstNode::from(2usize) + AstNode::from(1usize)
    );

    // Condition should be false
    let captured_nodes2_false = [AstNode::from(0usize), AstNode::from(2usize)];
    assert!(!(rule2.condition)(&captured_nodes2_false));
}
