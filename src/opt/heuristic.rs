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
}

impl<S: OptimizationSuggester, C: CostEstimator> HeuristicAstOptimizer<S, C> {
    pub fn new(suggester: S, cost_estimator: C) -> Self {
        Self {
            suggester,
            cost_estimator,
        }
    }
}

impl<S: OptimizationSuggester, C: CostEstimator> DeterministicAstOptimizer
    for HeuristicAstOptimizer<S, C>
{
    fn optimize(&self, node: AstNode) -> AstNode {
        // First, optimize children (post-order traversal)
        let new_src: Vec<AstNode> = node
            .src
            .iter()
            .map(|child| self.optimize(child.clone()))
            .collect();
        let mut best_node = AstNode::new(node.op.clone(), new_src, node.dtype.clone());

        // Generate suggestions for the current node
        let suggestions = self.suggester.suggest_optimizations(&best_node);

        if suggestions.is_empty() {
            return best_node;
        }

        let mut min_cost = self.cost_estimator.estimate_cost(&best_node);
        debug!(
            "Initial cost for node {:?}: {}",
            best_node.op, min_cost
        );

        // Evaluate suggestions and find the one with the minimum cost
        for suggestion in suggestions {
            let cost = self.cost_estimator.estimate_cost(&suggestion);
            debug!(
                "Cost for suggested node {:?}: {}",
                suggestion.op, cost
            );
            if cost < min_cost {
                min_cost = cost;
                best_node = suggestion;
                debug!("New best node found: {:?}", best_node.op);
            }
        }

        best_node
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
    fn test_heuristic_optimizer_with_distributive_law() {
        setup_logger();
        let a = AstNode::var("a");
        let b = AstNode::var("b");
        let c = AstNode::var("c");

        // Expression to optimize: a * (b + c)
        // This has 4 nodes: Mul, Var(a), Add, Var(b), Var(c) -> Cost = 5
        let target = a.clone() * (b.clone() + c.clone());

        // Expected result: a * b + a * c
        // This has 5 nodes: Add, Mul, Var(a), Var(b), Mul, Var(a), Var(c) -> Cost = 7
        let expected = (a.clone() * b.clone()) + (a.clone() * c.clone());

        // The distributive law rule
        let rule = rule!("distributive", |x, y, z| x.clone() * (y.clone() + z.clone()) => (x.clone() * y) + (x * z));
        let suggester = RuleBasedSuggester::new(vec![rule]);
        let cost_estimator = NodeCountCostEstimator;
        let optimizer = HeuristicAstOptimizer::new(suggester, cost_estimator);

        let result = optimizer.optimize(target.clone());

        // Since the cost of the rewritten expression (7) is higher than the original (5),
        // the optimizer should choose to NOT apply the rule.
        assert_eq!(result, target);

        // Now, let's test the opposite: a * b + a * c -> a * (b + c)
        let reversed_rule = rule!("factorization", |x, y, z| (x.clone() * y.clone()) + (x.clone() * z.clone()) => x * (y + z));
        let suggester_reversed = RuleBasedSuggester::new(vec![reversed_rule]);
        let optimizer_reversed = HeuristicAstOptimizer::new(suggester_reversed, cost_estimator);

        let result_reversed = optimizer_reversed.optimize(expected.clone());

        // The cost of the rewritten expression (5) is lower than the original (7),
        // so the optimizer SHOULD apply the rule.
        assert_eq!(result_reversed, target);
    }
}
