use crate::ast::AstNode;
use std::ops::Add;
use std::rc::Rc;

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
            pattern,
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
/// let rule = astpat!(|a| a + 1isize => a);
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

#[macro_export]
macro_rules! ast_rewriter {
    ($name:expr, $($rule:expr),*) => {
        AstRewriter::with_rules($name, vec![$($rule),*])
    };
}

pub struct AstRewriter {
    #[allow(dead_code)]
    name: String,
    rules: Vec<Rc<RewriteRule>>,
}

impl AstRewriter {
    #[allow(dead_code)]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            rules: Vec::new(),
        }
    }

    pub fn with_rules(name: &str, rules: Vec<Rc<RewriteRule>>) -> Self {
        Self {
            name: name.to_string(),
            rules,
        }
    }

    #[allow(dead_code)]
    pub fn add_rule(&mut self, rule: Rc<RewriteRule>) {
        self.rules.push(rule);
    }

    pub fn rewrite(&self, node: &AstNode) -> AstNode {
        // First, rewrite the children (post-order traversal)
        let new_args: Vec<AstNode> = node.src.iter().map(|arg| self.rewrite(arg)).collect();
        let rewritten_node = if new_args != node.src {
            AstNode::_new(node.op.clone(), new_args, node.dtype.clone())
        } else {
            node.clone()
        };

        // Then, try to rewrite the current node
        for rule in &self.rules {
            if let Some(captures) = rewritten_node.matches(&rule.pattern)
                && (rule.condition)(&captures)
            {
                // If the pattern matches and the condition is met,
                // apply the rewriter and return the new node.
                return (rule.rewriter)(&captures);
            }
        }

        rewritten_node
    }
}

impl Add for AstRewriter {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let name = format!("{}+{}", self.name, rhs.name);
        let rules = self.rules.into_iter().chain(rhs.rules).collect();
        AstRewriter::with_rules(&name, rules)
    }
}

#[test]
fn test_astpat_macro() {
    use crate::ast::AstNode;

    // Test case 1: Simple rule without condition
    let rule1 = astpat!(|a| a => a);
    let captured_nodes1 = [AstNode::from(1isize)];
    assert!((rule1.condition)(&captured_nodes1)); // Should always be true
    assert_eq!((rule1.rewriter)(&captured_nodes1), AstNode::from(1isize));

    // Test case 2: Rule with a condition
    let rule2 = astpat!(|a, _b| a + _b, if a == AstNode::from(1isize) => _b + a);

    // Condition should be true
    let captured_nodes2_true = [AstNode::from(1isize), AstNode::from(2isize)];
    assert!((rule2.condition)(&captured_nodes2_true));
    assert_eq!(
        (rule2.rewriter)(&captured_nodes2_true),
        AstNode::from(2isize) + AstNode::from(1isize)
    );

    // Condition should be false
    let captured_nodes2_false = [AstNode::from(0isize), AstNode::from(2isize)];
    assert!(!(rule2.condition)(&captured_nodes2_false));
}

#[cfg(test)]
#[allow(unused_variables)]
mod rewriter_tests {
    use super::*;
    use crate::ast::AstNode;

    #[test]
    fn test_simple_rewrite() {
        // Rule: a + b => b + a
        let rule = astpat!(|a, b| a + b => b + a);
        let rewriter = AstRewriter::with_rules("CommutativeAddition", vec![rule]);

        let expr = AstNode::from(1isize) + AstNode::from(2isize);
        let rewritten_expr = rewriter.rewrite(&expr);

        let expected = AstNode::from(2isize) + AstNode::from(1isize);
        assert_eq!(rewritten_expr, expected);
    }

    #[test]
    fn test_conditional_rewrite() {
        // Rule: a + 0 => a
        let rule = astpat!(|a, b| a + b, if b == AstNode::from(0isize) => a);
        let rewriter = AstRewriter::with_rules("AddZero", vec![rule]);

        // This should be rewritten
        let expr1 = AstNode::from(5isize) + AstNode::from(0isize);
        let rewritten_expr1 = rewriter.rewrite(&expr1);
        assert_eq!(rewritten_expr1, AstNode::from(5isize));

        // This should not be rewritten
        let expr2 = AstNode::from(5isize) + AstNode::from(1isize);
        let rewritten_expr2 = rewriter.rewrite(&expr2);
        assert_eq!(rewritten_expr2, expr2);
    }

    #[test]
    fn test_recursive_rewrite() {
        // Rule: a + 0 => a
        let rule = astpat!(|a, b| a + b, if b == AstNode::from(0isize) => a);
        let rewriter = AstRewriter::with_rules("AddZero", vec![rule]);

        // (1 + 0) + (2 + 0)
        let expr = (AstNode::from(1isize) + AstNode::from(0isize))
            + (AstNode::from(2isize) + AstNode::from(0isize));

        let rewritten_expr = rewriter.rewrite(&expr);

        // expected: 1 + 2
        let expected = AstNode::from(1isize) + AstNode::from(2isize);
        assert_eq!(rewritten_expr, expected);
    }

    #[test]
    fn test_no_rewrite() {
        let rule = astpat!(|a| a * AstNode::from(2isize) => a.clone() + a);
        let rewriter = AstRewriter::with_rules("MulToAd", vec![rule]);

        let expr = AstNode::from(1isize) + AstNode::from(2isize);
        let rewritten_expr = rewriter.rewrite(&expr);

        assert_eq!(expr, rewritten_expr);
    }

    #[test]
    fn test_add_rewriter() {
        // Rule 1: a + 0 => a
        let add_zero_rule = astpat!(|a, b| a + b, if b == AstNode::from(0isize) => a);
        let add_zero_rewriter = AstRewriter::with_rules("AddZero", vec![add_zero_rule]);

        // Rule 2: a * 1 => a
        let mul_one_rule = astpat!(|a, b| a * b, if b == AstNode::from(1isize) => a);
        let mul_one_rewriter = AstRewriter::with_rules("MulOne", vec![mul_one_rule]);

        let combined_rewriter = add_zero_rewriter + mul_one_rewriter;

        // Test AddZero rule
        let expr1 = AstNode::from(5isize) + AstNode::from(0isize);
        let rewritten_expr1 = combined_rewriter.rewrite(&expr1);
        assert_eq!(rewritten_expr1, AstNode::from(5isize));

        // Test MulOne rule
        let expr2 = AstNode::from(5isize) * AstNode::from(1isize);
        let rewritten_expr2 = combined_rewriter.rewrite(&expr2);
        assert_eq!(rewritten_expr2, AstNode::from(5isize));

        // Test both
        let expr3 = (AstNode::from(5isize) + AstNode::from(0isize)) * AstNode::from(1isize);
        let rewritten_expr3 = combined_rewriter.rewrite(&expr3);
        assert_eq!(rewritten_expr3, AstNode::from(5isize));
    }

    #[test]
    fn test_ast_rewriter_macro() {
        let rule1 = astpat!(|a, b| a + b => b + a);
        let rule2 = astpat!(|a, b| a * b => b * a);
        let rewriter = ast_rewriter!("Commutative", rule1, rule2);

        assert_eq!(rewriter.name, "Commutative");
        assert_eq!(rewriter.rules.len(), 2);
    }

    #[test]
    fn test_matches_simple() {
        let a = AstNode::capture(0);
        let pattern = a + AstNode::from(1isize);
        let expr = AstNode::from(10isize) + AstNode::from(1isize);
        let captures = expr.matches(&pattern).unwrap();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0], AstNode::from(10isize));
    }

    #[test]
    fn test_matches_multiple_captures() {
        let a = AstNode::capture(0);
        let b = AstNode::capture(1);
        let pattern = a + b;
        let expr = AstNode::from(10isize) + AstNode::from(20isize);
        let captures = expr.matches(&pattern).unwrap();
        assert_eq!(captures.len(), 2);
        assert_eq!(captures[0], AstNode::from(10isize));
        assert_eq!(captures[1], AstNode::from(20isize));
    }

    #[test]
    fn test_matches_no_match() {
        let a = AstNode::capture(0);
        let pattern = a + AstNode::from(1isize);
        let expr = AstNode::from(10isize) * AstNode::from(1isize);
        assert!(expr.matches(&pattern).is_none());
    }

    #[test]
    fn test_matches_nested() {
        let a = AstNode::capture(0);
        let pattern = (a + AstNode::from(1isize)) * AstNode::from(2isize);
        let expr = (AstNode::from(10isize) + AstNode::from(1isize)) * AstNode::from(2isize);
        let captures = expr.matches(&pattern).unwrap();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0], AstNode::from(10isize));
    }
}
