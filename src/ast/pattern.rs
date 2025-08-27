use crate::ast::AstNode;
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
        // First, apply the rewrite to all children of the current node.
        let children: Vec<AstNode> = ast
            .children()
            .into_iter()
            .map(|child| self.apply_recursive(child))
            .collect();

        // Rebuild the current node with the rewritten children.
        let new_ast = ast.replace_children(children);

        // Then, try to find rewrites for the current node.
        // If there are possible rewrites, apply the first one.
        // Otherwise, return the node as is.
        self.get_possible_rewrites(&new_ast)
            .first()
            .cloned()
            .unwrap_or(new_ast)
    }

    pub fn get_possible_rewrites(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut rewrites = vec![];

        // Try to match the current node.
        if let Some(captured_options) = self.match_node(ast, &self.pattern) {
            // All captures must be Some(...) for a match to be valid.
            if let Some(captured_refs) = captured_options.into_iter().collect::<Option<Vec<_>>>() {
                let captured_nodes: Vec<AstNode> = captured_refs.into_iter().cloned().collect();
                if (self.condition)(&captured_nodes) {
                    rewrites.push((self.rewriter)(&captured_nodes));
                }
            }
        }

        // Recursively find rewrites in children.
        for (i, child) in ast.children().iter().enumerate() {
            for rewritten_child in self.get_possible_rewrites(child) {
                let mut new_children = ast.children().into_iter().cloned().collect::<Vec<_>>();
                new_children[i] = rewritten_child;
                rewrites.push(ast.replace_children(new_children));
            }
        }

        rewrites
    }

    fn match_node<'a>(
        &self,
        ast: &'a AstNode,
        pattern: &AstNode,
    ) -> Option<Vec<Option<&'a AstNode>>> {
        match pattern {
            AstNode::Capture(n) => {
                let mut captured = vec![None; *n + 1];
                captured[*n] = Some(ast);
                Some(captured)
            }
            _ => {
                if std::mem::discriminant(ast) == std::mem::discriminant(pattern) {
                    let ast_children = ast.children();
                    let pattern_children = pattern.children();
                    if ast_children.len() == pattern_children.len() {
                        let mut all_captures: Vec<Option<&'a AstNode>> = vec![];
                        for (ac, pc) in ast_children.iter().zip(pattern_children.iter()) {
                            if let Some(child_captures) = self.match_node(ac, pc) {
                                // Merge captures from children.
                                for (i, capture) in child_captures.iter().enumerate() {
                                    if i >= all_captures.len() {
                                        all_captures.resize(i + 1, None);
                                    }
                                    if capture.is_some() {
                                        // It's possible for a capture to be overwritten if the pattern is weird,
                                        // but the macro should prevent this.
                                        all_captures[i] = *capture;
                                    }
                                }
                            } else {
                                return None;
                            }
                        }
                        Some(all_captures)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
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
