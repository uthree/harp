use std::rc::Rc;

use crate::ast::AstNode;

pub struct RewriteRule {
    pattern: AstNode,
    rewriter: Box<dyn Fn(&[AstNode]) -> AstNode>,
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

macro_rules! astpat {
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
    let pat = astpat!(|a| a => a);
}
