use crate::ast::AstNode;
use crate::opt::ast::heuristic::{CostEstimator, RewriteSuggester};
use console::Style;
use indicatif::HumanDuration;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::time::Instant;

/// An optimizer that uses beam search to find a low-cost AST.
pub struct BeamSearchAstOptimizer<S: RewriteSuggester, C: CostEstimator> {
    suggester: S,
    cost_estimator: C,
    pub beam_width: usize,
    pub max_steps: usize,
}

impl<S: RewriteSuggester, C: CostEstimator> BeamSearchAstOptimizer<S, C> {
    pub fn new(suggester: S, cost_estimator: C) -> Self {
        Self {
            suggester,
            cost_estimator,
            beam_width: 4,  // Set default beam width to 4
            max_steps: 100, // Default max steps for the search
        }
    }

    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn optimize(&self, initial_node: &AstNode) -> AstNode {
        let mut beam = vec![(
            initial_node.clone(),
            self.cost_estimator.estimate_cost(initial_node),
        )];
        let mut visited = HashSet::new();
        visited.insert(initial_node.clone());

        // Create a progress bar to visualize the optimization process.
        let start = Instant::now();
        let pb = ProgressBar::new(self.max_steps as u64);
        pb.set_style(
            ProgressStyle::with_template("{prefix:>12.cyan.bold} [{bar:24}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_prefix("Optimizing");

        for _ in 0..self.max_steps {
            let mut candidates = Vec::new();
            for (node, _) in &beam {
                let suggestions = self.suggester.suggest(node);
                for suggestion in suggestions {
                    if !visited.contains(&suggestion) {
                        let cost = self.cost_estimator.estimate_cost(&suggestion);
                        candidates.push((suggestion.clone(), cost));
                        visited.insert(suggestion);
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            beam = candidates.into_iter().take(self.beam_width).collect();
            pb.inc(1);
            pb.tick();
        }

        pb.finish_and_clear();
        let green_bold = Style::new().green().bold();
        pb.println(format!(
            "{:>12} optimize AST with beam search algorithm in {}",
            green_bold.apply_to("Finished"),
            HumanDuration(start.elapsed())
        ));

        beam.into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstOp, DType};
    use crate::{ast_rewriter, astpat};
    use std::collections::HashMap;

    struct MockCostEstimator {
        costs: HashMap<AstNode, f32>,
    }

    impl CostEstimator for MockCostEstimator {
        fn estimate_cost(&self, ast: &AstNode) -> f32 {
            self.costs.get(ast).copied().unwrap_or(999.0)
        }
    }

    struct MockRewriteSuggester {
        rewriter: crate::ast::pattern::AstRewriter,
    }

    impl RewriteSuggester for MockRewriteSuggester {
        fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
            self.rewriter.get_possible_rewrites(node)
        }
    }

    #[test]
    fn test_beam_search_optimizer() {
        let a = AstNode::var("a", DType::Isize);
        let zero = AstNode::from(0isize);

        // Expressions
        let expr_initial = a.clone() + zero.clone(); // a + 0
        let expr_optimized = a.clone(); // a

        // Cost Estimator: 'a' is cheaper than 'a + 0'
        let mut costs = HashMap::new();
        costs.insert(expr_initial.clone(), 10.0);
        costs.insert(expr_optimized.clone(), 1.0);
        let cost_estimator = MockCostEstimator { costs };

        // Suggester: knows how to rewrite 'a + 0' to 'a'
        let rule = astpat!(|x, y| x + y, if y.op == AstOp::Const(crate::ast::Const::Isize(0)) => x);
        let rewriter = ast_rewriter!("AddZero", rule);
        let suggester = MockRewriteSuggester { rewriter };

        // Optimizer
        let optimizer = BeamSearchAstOptimizer::new(suggester, cost_estimator)
            .with_beam_width(1)
            .with_max_steps(10);

        let optimized_node = optimizer.optimize(&expr_initial);

        assert_eq!(optimized_node, expr_optimized);
    }
}
