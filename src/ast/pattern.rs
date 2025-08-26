use crate::ast::AstNode;
use std::ops::Add;
use std::rc::Rc;

pub struct AstRewriteRule {
    pattern: AstNode,
    rewriter: Box<dyn Fn(&[AstNode]) -> AstNode>,
    condition: Box<dyn Fn(&[AstNode]) -> bool>,
}

impl AstRewriteRule {
    pub fn new(
        pattern: AstNode,
        rewriter: impl Fn(&[AstNode]) -> AstNode + 'static,
        condition: impl Fn(&[AstNode]) -> bool + 'static,
    ) -> Rc<Self> {
        Rc::new(AstRewriteRule {
            pattern,
            rewriter: Box::new(rewriter),
            condition: Box::new(condition),
        })
    }

    pub fn apply_recursive(&self, ast: &AstNode) -> AstNode {
        todo!()
    }

    pub fn get_possible_rewrites(&self, ast: &AstNode) -> Vec<AstNode> {
        todo!()
    }
}

/// A macro to create a `AstRewriteRule`.
///
/// # Example
///
/// The following example is ignored because `ast_pattern!` is not exported.
/// ```
/// use harp::ast_pattern;
///
/// // without condition
/// let rule = ast_pattern!(|a| a + 1isize => a.clone());
///
/// // with condition
/// let rule = ast_pattern!(|a, b| a + b, if *a == *b => b + a);
/// ```
#[macro_export]
macro_rules! ast_pattern {
    (| $($capture: pat_param),* | $pattern: expr, if $condition: expr => $rewriter: expr) => {
        {
            let mut counter = 0..;
            $(
                let $capture = $crate::ast::AstNode::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = |captured_nodes: &[$crate::ast::AstNode]| {
                let mut counter = 0..;
                $(
                    let $capture = &captured_nodes[counter.next().unwrap()];
                )*
                $rewriter
            };
            let condition = |captured_nodes: &[$crate::ast::AstNode]| {
                let mut counter = 0..;
                $(
                    let $capture = &captured_nodes[counter.next().unwrap()];
                )*
                $condition
            };
            $crate::ast::pattern::AstRewriteRule::new(pattern, rewriter, condition)
        }
    };
    (| $($capture: pat_param),* | $pattern: expr => $rewriter: expr ) => {
        {
            let mut counter = 0..;
            $(
                let $capture = $crate::ast::AstNode::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = |captured_nodes: &[$crate::ast::AstNode]| {
                let mut counter = 0..;
                $(
                    let $capture = &captured_nodes[counter.next().unwrap()];
                )*
                $rewriter
            };
            $crate::ast::pattern::AstRewriteRule::new(pattern, rewriter, |_| true)
        }
    };
}

#[derive(Clone)]
pub struct AstRewriter {
    #[allow(dead_code)]
    name: String,
    rules: Vec<Rc<AstRewriteRule>>,
}

#[macro_export]
macro_rules! ast_rewriter {
    ($name:expr, $($rule:expr),*) => {
        $crate::ast::pattern::AstRewriter::new($name, vec![$($rule),*])
    };
}

impl AstRewriter {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            rules: Vec::new(),
        }
    }

    pub fn with_rules(name: &str, rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self {
            name: name.to_string(),
            rules,
        }
    }

    pub fn add_rule(&mut self, rule: Rc<AstRewriteRule>) {
        self.rules.push(rule);
    }
}
