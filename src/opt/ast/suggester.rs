use crate::ast::AstNode;
use crate::opt::ast::heuristic::{CostEstimator, RewriteSuggester};

// Re-export suggester and cost estimator implementations from heuristic module
pub use crate::opt::ast::heuristic::{
    CommutativeSuggester, LoopTransformSuggester, NodeCountCostEstimator, OperationCostEstimator,
    RedundancyRemovalSuggester, RuleBasedSuggester,
};

/// An optimizer that uses a cost estimator to select the best rewrite.
pub struct CostBasedOptimizer<S: RewriteSuggester, E: CostEstimator> {
    suggester: S,
    estimator: E,
    max_iterations: usize,
    show_progress: bool,
}

impl<S: RewriteSuggester, E: CostEstimator> CostBasedOptimizer<S, E> {
    pub fn new(suggester: S, estimator: E, max_iterations: usize) -> Self {
        Self {
            suggester,
            estimator,
            max_iterations,
            show_progress: false,
        }
    }

    pub fn with_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    pub fn optimize(&self, ast: &AstNode) -> AstNode {
        use indicatif::{ProgressBar, ProgressStyle};

        let mut current = ast.clone();
        let mut current_cost = self.estimator.estimate_cost(&current);

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_iterations as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_message(format!("Cost: {:.2}", current_cost));
            Some(pb)
        } else {
            None
        };

        for i in 0..self.max_iterations {
            let suggestions = self.suggester.suggest(&current);
            if suggestions.is_empty() {
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_iterations as u64);
                    pb.finish_with_message(format!(
                        "Completed (no more suggestions) - Final cost: {:.2}",
                        current_cost
                    ));
                }
                break;
            }

            // Find the best suggestion
            let mut best = current.clone();
            let mut best_cost = current_cost;

            for suggestion in suggestions {
                let cost = self.estimator.estimate_cost(&suggestion);
                if cost < best_cost {
                    best = suggestion;
                    best_cost = cost;
                }
            }

            // If no improvement, stop
            if best_cost >= current_cost {
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_iterations as u64);
                    pb.finish_with_message(format!(
                        "Completed (no improvement) - Final cost: {:.2}",
                        current_cost
                    ));
                }
                break;
            }

            current = best;
            current_cost = best_cost;

            if let Some(ref pb) = pb {
                pb.set_position((i + 1) as u64);
                pb.set_message(format!("Cost: {:.2}", current_cost));
            }
        }

        if let Some(ref pb) = pb {
            if pb.position() < self.max_iterations as u64 {
                pb.set_position(self.max_iterations as u64);
                pb.finish_with_message(format!("Final cost: {:.2}", current_cost));
            }
        }

        current
    }
}

/// An optimizer that uses beam search to explore multiple optimization paths.
///
/// Beam search maintains a set of k best candidates (the "beam") at each iteration,
/// allowing it to explore a wider search space than greedy optimization.
pub struct BeamSearchOptimizer<S: RewriteSuggester, E: CostEstimator> {
    suggester: S,
    estimator: E,
    beam_width: usize,
    max_iterations: usize,
    show_progress: bool,
}

impl<S: RewriteSuggester, E: CostEstimator> BeamSearchOptimizer<S, E> {
    pub fn new(suggester: S, estimator: E, beam_width: usize, max_iterations: usize) -> Self {
        Self {
            suggester,
            estimator,
            beam_width,
            max_iterations,
            show_progress: false,
        }
    }

    pub fn with_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    pub fn optimize(&self, ast: &AstNode) -> AstNode {
        use indicatif::{ProgressBar, ProgressStyle};
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        // Wrapper to make AstNode sortable by cost
        #[derive(Clone)]
        struct Candidate {
            ast: AstNode,
            cost: f32,
        }

        impl PartialEq for Candidate {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }

        impl Eq for Candidate {}

        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // Reverse ordering for min-heap
                other.cost.partial_cmp(&self.cost)
            }
        }

        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        // Initialize beam with the original AST
        let initial_cost = self.estimator.estimate_cost(ast);
        let mut beam = vec![Candidate {
            ast: ast.clone(),
            cost: initial_cost,
        }];

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_iterations as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_message(format!(
                "Beam size: {}, Best cost: {:.2}",
                beam.len(),
                initial_cost
            ));
            Some(pb)
        } else {
            None
        };

        for i in 0..self.max_iterations {
            let mut candidates = BinaryHeap::new();

            // Generate all possible rewrites from current beam
            for current in &beam {
                let suggestions = self.suggester.suggest(&current.ast);

                for suggestion in suggestions {
                    let cost = self.estimator.estimate_cost(&suggestion);
                    candidates.push(Candidate {
                        ast: suggestion,
                        cost,
                    });
                }
            }

            // If no new candidates, we're done
            if candidates.is_empty() {
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_iterations as u64);
                    let best_cost = beam
                        .iter()
                        .map(|c| c.cost)
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                        .unwrap_or(initial_cost);
                    pb.finish_with_message(format!(
                        "Completed (no more candidates) - Final cost: {:.2}",
                        best_cost
                    ));
                }
                break;
            }

            // Select top k candidates for the new beam
            beam.clear();
            for _ in 0..self.beam_width {
                if let Some(candidate) = candidates.pop() {
                    beam.push(candidate);
                } else {
                    break;
                }
            }

            // If beam is empty, we're stuck
            if beam.is_empty() {
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_iterations as u64);
                    pb.finish_with_message(format!(
                        "Completed (beam empty) - Final cost: {:.2}",
                        initial_cost
                    ));
                }
                break;
            }

            if let Some(ref pb) = pb {
                pb.set_position((i + 1) as u64);
                let best_cost = beam
                    .iter()
                    .map(|c| c.cost)
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(initial_cost);
                pb.set_message(format!(
                    "Beam size: {}, Best cost: {:.2}",
                    beam.len(),
                    best_cost
                ));
            }
        }

        let best = beam
            .into_iter()
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal))
            .map(|c| c.ast)
            .unwrap_or_else(|| ast.clone());

        if let Some(ref pb) = pb {
            let final_cost = self.estimator.estimate_cost(&best);
            if pb.position() < self.max_iterations as u64 {
                pb.set_position(self.max_iterations as u64);
                pb.finish_with_message(format!("Final cost: {:.2}", final_cost));
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i(val: isize) -> AstNode {
        AstNode::from(val)
    }

    #[test]
    fn test_commutative_suggester() {
        let suggester = CommutativeSuggester;
        let ast = i(1) + i(2);
        let suggestions = suggester.suggest(&ast);

        // Should suggest swapping the operands
        assert!(!suggestions.is_empty());
        assert!(suggestions.contains(&(i(2) + i(1))));
    }

    #[test]
    fn test_node_count_cost_estimator() {
        let estimator = NodeCountCostEstimator;
        let ast1 = i(1);
        let ast2 = i(1) + i(2);
        let ast3 = (i(1) + i(2)) * i(3);

        assert!(estimator.estimate_cost(&ast1) < estimator.estimate_cost(&ast2));
        assert!(estimator.estimate_cost(&ast2) < estimator.estimate_cost(&ast3));
    }

    #[test]
    fn test_operation_cost_estimator() {
        let estimator = OperationCostEstimator;
        let add = i(1) + i(2);
        let mul = i(1) * i(2);
        let div = i(1) / i(2);

        // Division should be more expensive than multiplication
        assert!(estimator.estimate_cost(&div) > estimator.estimate_cost(&mul));
        // Multiplication should be more expensive than addition
        assert!(estimator.estimate_cost(&mul) > estimator.estimate_cost(&add));
    }

    #[test]
    fn test_rule_based_suggester() {
        use crate::ast_pattern;

        // Create a simple rule: a * 2 -> a + a
        let rule = ast_pattern!(|a| a * i(2) => a.clone() + a.clone());
        let suggester = RuleBasedSuggester::new(vec![rule]);

        // Check if the suggester generates suggestions
        let ast = AstNode::Var("a".to_string()) * i(2);
        let suggestions = suggester.suggest(&ast);

        // Should suggest at least one rewrite
        assert!(!suggestions.is_empty());
        assert!(
            suggestions.contains(&(AstNode::Var("a".to_string()) + AstNode::Var("a".to_string())))
        );
    }

    #[test]
    fn test_cost_based_optimizer() {
        use crate::ast_pattern;

        // Create a simple rule: a * 2 -> a + a
        let rule = ast_pattern!(|a| a * i(2) => a.clone() + a.clone());
        let suggester = RuleBasedSuggester::new(vec![rule]);
        let estimator = OperationCostEstimator;
        let optimizer = CostBasedOptimizer::new(suggester, estimator, 10);

        // a * 2 should be optimized to a + a (cheaper)
        let ast = AstNode::Var("a".to_string()) * i(2);
        let optimized = optimizer.optimize(&ast);

        // Check if optimization was applied
        // Note: The optimizer only applies if the cost is lower
        // Addition has cost 1, Multiplication has cost 2
        let expected = AstNode::Var("a".to_string()) + AstNode::Var("a".to_string());
        let original_cost = estimator.estimate_cost(&ast);
        let expected_cost = estimator.estimate_cost(&expected);

        // If expected cost is lower, the optimization should be applied
        if expected_cost < original_cost {
            assert_eq!(optimized, expected);
        } else {
            // Otherwise, the original AST should be returned
            assert_eq!(optimized, ast);
        }
    }

    #[test]
    fn test_beam_search_optimizer_simple() {
        use crate::ast_pattern;

        // Create a simple rule: a * 2 -> a + a
        let rule = ast_pattern!(|a| a * i(2) => a.clone() + a.clone());
        let suggester = RuleBasedSuggester::new(vec![rule]);
        let estimator = OperationCostEstimator;
        let optimizer = BeamSearchOptimizer::new(suggester, estimator, 3, 10);

        // a * 2 should be optimized to a + a (cheaper)
        let ast = AstNode::Var("a".to_string()) * i(2);
        let optimized = optimizer.optimize(&ast);

        let expected = AstNode::Var("a".to_string()) + AstNode::Var("a".to_string());
        let original_cost = estimator.estimate_cost(&ast);
        let optimized_cost = estimator.estimate_cost(&optimized);

        // The optimized version should have equal or lower cost
        assert!(optimized_cost <= original_cost);

        // If a better solution exists, it should find it
        let expected_cost = estimator.estimate_cost(&expected);
        if expected_cost < original_cost {
            assert_eq!(optimized, expected);
        }
    }

    #[test]
    fn test_beam_search_optimizer_multiple_paths() {
        use crate::ast_pattern;

        // Create multiple rules
        let rule1 = ast_pattern!(|a| a * i(2) => a.clone() + a.clone());
        let rule2 = ast_pattern!(|a| a + i(0) => a.clone());
        let suggester = RuleBasedSuggester::new(vec![rule1, rule2]);
        let estimator = NodeCountCostEstimator;
        let optimizer = BeamSearchOptimizer::new(suggester, estimator, 5, 10);

        // (a * 2) + 0 should be optimized
        let ast = (AstNode::Var("a".to_string()) * i(2)) + i(0);
        let optimized = optimizer.optimize(&ast);

        // Should be smaller than the original
        let original_cost = estimator.estimate_cost(&ast);
        let optimized_cost = estimator.estimate_cost(&optimized);
        assert!(optimized_cost <= original_cost);
    }

    #[test]
    fn test_beam_search_vs_greedy() {
        use crate::ast_pattern;

        // Create rules that might lead to different optimal paths
        let rule1 = ast_pattern!(|a, b| a + b => b.clone() + a.clone()); // Commutative
        let rule2 = ast_pattern!(|a| a * i(1) => a.clone());
        let suggester = RuleBasedSuggester::new(vec![rule1, rule2]);
        let estimator = NodeCountCostEstimator;

        let beam_optimizer = BeamSearchOptimizer::new(suggester.clone(), estimator, 10, 5);
        let greedy_optimizer = CostBasedOptimizer::new(suggester, estimator, 5);

        let ast = (i(1) * AstNode::Var("x".to_string())) + AstNode::Var("y".to_string());

        let beam_result = beam_optimizer.optimize(&ast);
        let greedy_result = greedy_optimizer.optimize(&ast);

        // Both should produce valid optimizations
        let original_cost = estimator.estimate_cost(&ast);
        let beam_cost = estimator.estimate_cost(&beam_result);
        let greedy_cost = estimator.estimate_cost(&greedy_result);

        assert!(beam_cost <= original_cost);
        assert!(greedy_cost <= original_cost);

        // Beam search should be at least as good as greedy
        assert!(beam_cost <= greedy_cost);
    }
}
