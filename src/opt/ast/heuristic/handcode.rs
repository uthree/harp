use crate::ast::pattern::{AstRewriteRule, AstRewriter};
use crate::ast::{AstNode, AstOp};
use crate::astpat;
use crate::opt::ast::heuristic::{CostEstimator, RewriteSuggester};
use crate::opt::ast::rule::algebraic_simplification;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub struct HandcodedCostEstimator;

impl CostEstimator for HandcodedCostEstimator {
    fn estimate_cost(&self, node: &AstNode) -> f32 {
        match &node.op {
            AstOp::Range { step, .. } => {
                if node.src.is_empty() {
                    return 0.0; // No loop bound or body
                }
                let max_node = &node.src[0];
                let body_nodes = &node.src[1..];

                let max_node_cost = self.estimate_cost(max_node);
                let body_cost: f32 = body_nodes.iter().map(|n| self.estimate_cost(n)).sum();
                let body_cost: f32 = body_cost + 5e-8f32; // The cost of loop counters required for iteration, conditional branches, and program counter movements

                let iterations = if let AstOp::Const(c) = &max_node.op {
                    if let Some(val) = c.as_isize() {
                        // Assuming start is 0 and step is a positive integer
                        (val as f32 / *step as f32).ceil().max(0.0)
                    } else {
                        100.0 // Non-integer constant for loop bound, use heuristic
                    }
                } else {
                    100.0 // Dynamic loop bound, use a heuristic multiplier
                };

                max_node_cost + (body_cost * iterations)
            }
            _ => {
                let op_cost = match &node.op {
                    AstOp::Div => 10.0,
                    AstOp::Store => 8.0,
                    AstOp::Assign => 5.0,
                    AstOp::Declare { .. } => 0.0,
                    AstOp::Func { .. } => 0.0,
                    AstOp::Var(_) => 2.0,
                    AstOp::Call(_) => 5.0,
                    _ => 1.0,
                } * 1e-9f32;
                let children_cost: f32 =
                    node.src.iter().map(|child| self.estimate_cost(child)).sum();
                op_cost + children_cost
            }
        }
    }
}

/// a * (b + c) => (a * b) + (a * c)
/// (a + b) * c => (a * c) + (b * c)
pub fn distributive_rules() -> AstRewriter {
    let rules: Vec<Rc<AstRewriteRule>> = vec![
        astpat!(|a, b, c| a * (b + c) => (a.clone() * b) + (a * c)),
        astpat!(|a, b, c| (a + b) * c => (a * c.clone()) + (b * c)),
    ];
    AstRewriter::with_rules("distributive rules", rules)
}

/// a + b => b + a
/// a * b => b * a
/// WARNING: These rules can cause infinite loops with the recursive `apply` method.
/// They are better suited for heuristic optimizers that can control application,
/// such as `get_possible_rewrites`.
pub fn commutative_rules() -> AstRewriter {
    let rules: Vec<Rc<AstRewriteRule>> = vec![
        astpat!(|a, b| a + b => b + a),
        astpat!(|a, b| a * b => b * a),
    ];
    AstRewriter::with_rules("commutative rules", rules)
}

/// (a + b) + c => a + (b + c)
/// (a * b) * c => a * (b * c)
pub fn associative_rules() -> AstRewriter {
    let rules: Vec<Rc<AstRewriteRule>> = vec![
        astpat!(|a, b, c| (a + b) + c => a + (b + c)),
        astpat!(|a, b, c| (a * b) * c => a * (b * c)),
    ];
    AstRewriter::with_rules("associative rules", rules)
}

pub struct AlgebraicSuggester {
    rewriter: AstRewriter,
}

impl Default for AlgebraicSuggester {
    fn default() -> Self {
        AlgebraicSuggester {
            rewriter: algebraic_simplification(),
        }
    }
}

impl RewriteSuggester for AlgebraicSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        self.rewriter.get_possible_rewrites(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    fn get_vars() -> (AstNode, AstNode, AstNode) {
        let a = AstNode::var("a", DType::Isize);
        let b = AstNode::var("b", DType::Isize);
        let c = AstNode::var("c", DType::Isize);
        (a, b, c)
    }

    #[test]
    fn test_distributive_rules() {
        let (a, b, c) = get_vars();
        let rewriter = distributive_rules();

        let expr1 = a.clone() * (b.clone() + c.clone());
        let expected1 = (a.clone() * b.clone()) + (a.clone() * c.clone());
        assert_eq!(rewriter.apply(&expr1), expected1);

        let expr2 = (a.clone() + b.clone()) * c.clone();
        let expected2 = (a.clone() * c.clone()) + (b.clone() * c.clone());
        assert_eq!(rewriter.apply(&expr2), expected2);
    }

    #[test]
    fn test_commutative_rules() {
        let (a, b, _) = get_vars();
        let rewriter = commutative_rules();

        let expr1 = a.clone() + b.clone();
        let expected1 = b.clone() + a.clone();
        assert_eq!(rewriter.apply(&expr1), expected1);

        let expr2 = a.clone() * b.clone();
        let expected2 = b.clone() * a.clone();
        assert_eq!(rewriter.apply(&expr2), expected2);
    }

    #[test]
    fn test_associative_rules() {
        let (a, b, c) = get_vars();
        let rewriter = associative_rules();

        let expr1 = (a.clone() + b.clone()) + c.clone();
        let expected1 = a.clone() + (b.clone() + c.clone());
        assert_eq!(rewriter.apply(&expr1), expected1);

        let expr2 = (a.clone() * b.clone()) * c.clone();
        let expected2 = a.clone() * (b.clone() * c.clone());
        assert_eq!(rewriter.apply(&expr2), expected2);
    }

    #[test]
    fn test_algebraic_suggester() {
        let suggester = AlgebraicSuggester::default();
        let expr = AstNode::var("a", DType::Isize) + AstNode::from(0isize);
        let suggestions = suggester.suggest(&expr);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0], AstNode::var("a", DType::Isize));
    }
}
