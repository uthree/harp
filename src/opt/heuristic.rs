use crate::ast::AstNode;
use crate::opt::ast::{CostEstimator, DeterministicAstOptimizer, OptimizationSuggester};
use log::debug;

/// A simple cost estimator that counts the number of nodes in the AST.
#[derive(Clone, Copy)]
pub struct NodeCountCostEstimator;

impl CostEstimator for NodeCountCostEstimator {
    fn estimate_cost(&self, node: &AstNode) -> f32 {
        let mut count = 1.0;
        for child in &node.src {
            count += self.estimate_cost(child);
        }
        count
    }
}

/// An optimizer that uses a suggester and a cost estimator to make decisions.
pub struct HeuristicAstOptimizer<S: OptimizationSuggester, C: CostEstimator> {
    suggester: S,
    cost_estimator: C,
    max_iterations: usize,
}

impl<S: OptimizationSuggester, C: CostEstimator> HeuristicAstOptimizer<S, C> {
    pub fn new(suggester: S, cost_estimator: C) -> Self {
        Self {
            suggester,
            cost_estimator,
            max_iterations: 100, // Default max iterations
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    fn apply_one_pass(&self, node: AstNode) -> AstNode {
        // First, optimize children (post-order traversal)
        let new_src: Vec<AstNode> = node
            .src
            .iter()
            .map(|child| self.apply_one_pass(child.clone()))
            .collect();
        let mut best_node = AstNode::new(node.op.clone(), new_src, node.dtype.clone());

        // Generate suggestions for the current node
        let suggestions = self.suggester.suggest_optimizations(&best_node);

        if suggestions.is_empty() {
            return best_node;
        }

        let mut min_cost = self.cost_estimator.estimate_cost(&best_node);
        debug!("Initial cost for node {:?}: {}", best_node.op, min_cost);

        // Evaluate suggestions and find the one with the minimum cost
        for suggestion in suggestions {
            let cost = self.cost_estimator.estimate_cost(&suggestion);
            debug!("Cost for suggested node {:?}: {}", suggestion.op, cost);
            if cost < min_cost {
                min_cost = cost;
                best_node = suggestion;
                debug!("New best node found: {:?}", best_node.op);
            }
        }

        best_node
    }
}

impl<S: OptimizationSuggester, C: CostEstimator> DeterministicAstOptimizer
    for HeuristicAstOptimizer<S, C>
{
    fn optimize(&self, mut node: AstNode) -> AstNode {
        for i in 0..self.max_iterations {
            let original_node = node.clone();
            let new_node = self.apply_one_pass(original_node.clone());

            if new_node == original_node {
                debug!("Greedy search reached fixed point after {i} iterations.");
                return new_node;
            }
            debug!("AST changed in iteration {i}. Continuing greedy search...");
            node = new_node;
        }
        debug!(
            "Greedy search finished after reaching max iterations ({}).",
            self.max_iterations
        );
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;
    use crate::opt::ast::{AstRewriter, RewriteRule};
    use crate::rule;
    use std::rc::Rc;

    fn setup_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    /// A suggester that uses a set of rewrite rules to generate optimization candidates.
    pub struct RuleBasedSuggester {
        rewriter: AstRewriter,
    }

    impl RuleBasedSuggester {
        pub fn new(rules: Vec<Rc<RewriteRule>>) -> Self {
            Self {
                rewriter: AstRewriter::new(rules),
            }
        }
    }

    impl OptimizationSuggester for RuleBasedSuggester {
        fn suggest_optimizations(&self, node: &AstNode) -> Vec<AstNode> {
            let rewritten_node = self.rewriter.apply(node.clone());
            if &rewritten_node != node {
                vec![rewritten_node]
            } else {
                vec![]
            }
        }
    }

    #[test]
    fn test_heuristic_optimizer_greedy_iteration() {
        setup_logger();
        let a = AstNode::var("a");
        let b = AstNode::var("b");
        let c = AstNode::var("c");
        let d = AstNode::var("d");
        let e = AstNode::var("e");
        let f = AstNode::var("f");

        // Expression to optimize: (a * b + a * c) + (d * e + d * f)
        // This is complex and has a high node count.
        let original_expr =
            (a.clone() * b.clone() + a.clone() * c.clone()) + (d.clone() * e.clone() + d.clone() * f.clone());

        // Expected result after factorization: a * (b + c) + d * (e + f)
        // This is simpler and has a lower node count.
        let expected_expr = (a.clone() * (b.clone() + c.clone())) + (d.clone() * (e.clone() + f.clone()));

        // Rule for factorization (a * b + a * c -> a * (b + c))
        let factorization_rule = rule!("factorization", |x, y, z| (x.clone() * y.clone()) + (x.clone() * z.clone()) => x * (y + z));
        let suggester = RuleBasedSuggester::new(vec![factorization_rule]);
        let cost_estimator = NodeCountCostEstimator;
        let optimizer = HeuristicAstOptimizer::new(suggester, cost_estimator);

        let result = optimizer.optimize(original_expr.clone());

        // The optimizer should iteratively apply the factorization rule
        // until it reaches the most simplified form.
        assert_eq!(result, expected_expr);

        // Let's also test the distributive law to ensure it does NOT get applied
        // because it increases the cost.
        let distributive_rule = rule!("distributive", |x, y, z| x.clone() * (y.clone() + z.clone()) => (x.clone() * y) + (x * z));
        let suggester_dist = RuleBasedSuggester::new(vec![distributive_rule]);
        let optimizer_dist = HeuristicAstOptimizer::new(suggester_dist, cost_estimator);

        let result_dist = optimizer_dist.optimize(expected_expr.clone());

        // The optimizer should not apply the distributive law as it increases the node count.
        assert_eq!(result_dist, expected_expr);
    }
}
