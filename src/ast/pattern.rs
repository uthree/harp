use crate::ast::AstNode;
use std::rc::Rc;

type RewriterFn = Box<dyn Fn(&[AstNode]) -> AstNode>;
type ConditionFn = Box<dyn Fn(&[AstNode]) -> bool>;

pub struct AstRewriteRule {
    pattern: AstNode,
    rewriter: RewriterFn,
    condition: ConditionFn,
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
        ast.clone().replace_if(
            |node| self.match_and_rewrite_top_level(node).is_some(),
            |node| self.match_and_rewrite_top_level(&node).unwrap(),
        )
    }

    pub fn get_possible_rewrites(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut rewrites = vec![];

        // Try to match the current node.
        if let Some(captured_options) = Self::match_node(ast, &self.pattern) {
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
                rewrites.push(ast.clone().replace_children(new_children));
            }
        }

        rewrites
    }

    /// Matches and rewrites only the top-level node, without recursing into children.
    fn match_and_rewrite_top_level(&self, ast: &AstNode) -> Option<AstNode> {
        if let Some(captured_options) = Self::match_node(ast, &self.pattern) {
            if let Some(captured_refs) = captured_options.into_iter().collect::<Option<Vec<_>>>() {
                let captured_nodes: Vec<AstNode> = captured_refs.into_iter().cloned().collect();
                if (self.condition)(&captured_nodes) {
                    return Some((self.rewriter)(&captured_nodes));
                }
            }
        }
        None
    }

    fn match_node<'a>(ast: &'a AstNode, pattern: &AstNode) -> Option<Vec<Option<&'a AstNode>>> {
        match pattern {
            AstNode::Capture(n) => {
                let mut captured = vec![None; *n + 1];
                captured[*n] = Some(ast);
                Some(captured)
            }
            AstNode::Const(p_val) => {
                if let AstNode::Const(a_val) = ast {
                    if a_val == p_val {
                        return Some(vec![]);
                    }
                }
                None
            }
            _ => {
                if std::mem::discriminant(ast) == std::mem::discriminant(pattern) {
                    let ast_children = ast.children();
                    let pattern_children = pattern.children();
                    if ast_children.len() == pattern_children.len() {
                        let mut all_captures: Vec<Option<&'a AstNode>> = vec![];
                        for (ac, pc) in ast_children.iter().zip(pattern_children.iter()) {
                            if let Some(child_captures) = Self::match_node(ac, pc) {
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
        $crate::ast::pattern::AstRewriter::with_rules($name.to_string(), vec![$($rule),*])
    };
}

impl AstRewriter {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            rules: Vec::new(),
        }
    }

    pub fn with_rules(name: String, rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self {
            name: name.to_string(),
            rules,
        }
    }

    pub fn add_rule(&mut self, rule: Rc<AstRewriteRule>) {
        self.rules.push(rule);
    }

    pub fn apply(&self, ast: &AstNode) -> AstNode {
        // Use the new replace_if functionality with repeated application
        let mut result = ast.clone();
        loop {
            let old_result = result.clone();
            result = result.replace_if(
                |node| {
                    self.rules
                        .iter()
                        .any(|rule| rule.match_and_rewrite_top_level(node).is_some())
                },
                |node| {
                    // Apply the first matching rule
                    for rule in &self.rules {
                        if let Some(rewritten) = rule.match_and_rewrite_top_level(&node) {
                            return rewritten;
                        }
                    }
                    node // This should never be reached due to the predicate
                },
            );

            if result == old_result {
                break; // No more changes
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::AstNode;
    use crate::{ast_pattern, ast_rewriter};

    fn i(val: isize) -> AstNode {
        AstNode::from(val)
    }

    #[test]
    fn test_simple_rewrite() {
        let rule = ast_pattern!(|a| a + i(0) => a.clone());
        let rewriter = ast_rewriter!("simple_arith", rule);

        let ast = i(1) + i(0);
        let rewritten_ast = rewriter.apply(&ast);
        assert_eq!(rewritten_ast, i(1));
    }

    #[test]
    fn test_recursive_rewrite() {
        let rule = ast_pattern!(|a| a + i(0) => a.clone());
        let rewriter = ast_rewriter!("simple_arith", rule);

        // Should apply to both children first, then the root
        let ast = (i(1) + i(0)) + (i(2) + i(0));
        let rewritten_ast = rewriter.apply(&ast);
        assert_eq!(rewritten_ast, i(1) + i(2));
    }

    #[test]
    fn test_repeated_rewrite() {
        let rule1 = ast_pattern!(|a| a * i(1) => a.clone());
        let rule2 = ast_pattern!(|a| a + i(0) => a.clone());
        let rewriter = ast_rewriter!("simplify", rule1, rule2);

        // (2 * 1) + 0  ->  2 + 0  ->  2
        let ast = (i(2) * i(1)) + i(0);
        let rewritten_ast = rewriter.apply(&ast);
        assert_eq!(rewritten_ast, i(2));
    }

    #[test]
    fn test_no_rewrite() {
        let rule = ast_pattern!(|a| a + i(0) => a.clone());
        let rewriter = ast_rewriter!("simple_arith", rule);

        let ast = i(1) + i(2);
        let original_ast_clone = ast.clone();
        let rewritten_ast = rewriter.apply(&ast);
        assert_eq!(rewritten_ast, original_ast_clone);
    }

    #[test]
    #[allow(unused_variables)]
    fn test_conditional_rewrite() {
        // Only rewrite a + a to 2 * a
        let rule = ast_pattern!(|a, b| a + b, if a == b => i(2) * a.clone());
        let rewriter = ast_rewriter!("simplify_add", rule);

        // This should be rewritten
        let ast1 = i(3) + i(3);
        let rewritten_ast1 = rewriter.apply(&ast1);
        assert_eq!(rewritten_ast1, i(2) * i(3));

        // This should not be rewritten
        let ast2 = i(3) + i(4);
        let original_ast2_clone = ast2.clone();
        let rewritten_ast2 = rewriter.apply(&ast2);
        assert_eq!(rewritten_ast2, original_ast2_clone);
    }
}
