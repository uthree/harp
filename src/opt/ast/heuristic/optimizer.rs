use crate::ast::AstNode;
use crate::opt::ast::heuristic::{CostEstimator, RewriteSuggester};

/// An optimizer that uses a cost estimator to select the best rewrite.
pub struct CostBasedOptimizer<S: RewriteSuggester, E: CostEstimator> {
    suggester: S,
    estimator: E,
    max_iterations: usize,
    max_history: usize,
    show_progress: bool,
}

impl<S: RewriteSuggester, E: CostEstimator> CostBasedOptimizer<S, E> {
    pub fn new(suggester: S, estimator: E, max_iterations: usize) -> Self {
        Self {
            suggester,
            estimator,
            max_iterations,
            // デフォルトでは最大10000件の履歴を保持（メモリ消費を制限）
            max_history: 10000,
            // DEBUGビルドの時は自動的にプログレスバーを有効化
            show_progress: cfg!(debug_assertions),
        }
    }

    pub fn with_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    pub fn optimize(&self, ast: &AstNode) -> AstNode {
        use indicatif::{ProgressBar, ProgressStyle};
        use std::collections::VecDeque;

        let mut current = ast.clone();
        let mut current_cost = self.estimator.estimate_cost(&current);

        // 千日手検出のため、訪問済みのASTノードを記録（VecDequeで履歴件数を制限）
        let mut visited = VecDeque::new();
        visited.push_back(format!("{:?}", current));

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_iterations as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{prefix:>12.cyan.bold} [{bar:24}] {pos}/{len} {wide_msg}")
                    .unwrap()
                    .progress_chars("=> "),
            );
            pb.set_prefix("Optimizing");
            pb.set_message(format!("cost {:.2}", current_cost));
            Some(pb)
        } else {
            None
        };

        for _i in 0..self.max_iterations {
            let suggestions = self.suggester.suggest(&current);
            if suggestions.is_empty() {
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_iterations as u64);
                    pb.finish_with_message(format!("cost {:.2} (converged)", current_cost));
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
                    pb.finish_with_message(format!("cost {:.2} (converged)", current_cost));
                }
                break;
            }

            // 千日手検出: 同じ状態が出現したら停止
            let best_repr = format!("{:?}", best);
            if visited.contains(&best_repr) {
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_iterations as u64);
                    pb.finish_with_message(format!("cost {:.2} (cycle detected)", current_cost));
                }
                break;
            }

            // 履歴に追加（上限に達したら古いものを削除）
            if visited.len() >= self.max_history {
                visited.pop_front();
            }
            visited.push_back(best_repr);

            current = best;
            current_cost = best_cost;

            if let Some(ref pb) = pb {
                pb.set_message(format!("cost {:.2}", current_cost));
                pb.inc(1);
            }
        }

        if let Some(ref pb) = pb {
            pb.finish_with_message(format!("cost {:.2}", current_cost));
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
    max_history: usize,
    show_progress: bool,
}

impl<S: RewriteSuggester, E: CostEstimator> BeamSearchOptimizer<S, E> {
    pub fn new(suggester: S, estimator: E, beam_width: usize, max_iterations: usize) -> Self {
        Self {
            suggester,
            estimator,
            beam_width,
            max_iterations,
            // デフォルトでは最大100件の履歴を保持（メモリ消費を制限）
            max_history: 100,
            // DEBUGビルドの時は自動的にプログレスバーを有効化
            show_progress: cfg!(debug_assertions),
        }
    }

    pub fn with_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    pub fn optimize(&self, ast: &AstNode) -> AstNode {
        use indicatif::{ProgressBar, ProgressStyle};
        use std::cmp::Ordering;
        use std::collections::{BinaryHeap, VecDeque};

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

        // 千日手検出のため、訪問済みのASTノードを記録（VecDequeで履歴件数を制限）
        let mut visited = VecDeque::new();
        visited.push_back(format!("{:?}", ast));

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_iterations as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{prefix:>12.cyan.bold} [{bar:24}] {pos}/{len} {wide_msg}")
                    .unwrap()
                    .progress_chars("=> "),
            );
            pb.set_prefix("Optimizing");
            pb.set_message(format!("beam {}, cost {:.2}", beam.len(), initial_cost));
            pb.tick();
            Some(pb)
        } else {
            None
        };

        for _i in 0..self.max_iterations {
            let mut candidates = BinaryHeap::new();

            // Generate all possible rewrites from current beam
            for current in &beam {
                let suggestions = self.suggester.suggest(&current.ast);

                for suggestion in suggestions {
                    // 千日手検出: すでに訪問済みのノードは候補に追加しない
                    let suggestion_repr = format!("{:?}", suggestion);
                    if visited.contains(&suggestion_repr) {
                        continue;
                    }

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
                    pb.finish_with_message(format!("cost {:.2} (converged)", best_cost));
                    pb.tick()
                }
                break;
            }

            // Select top k candidates for the new beam
            beam.clear();
            for _ in 0..self.beam_width {
                if let Some(candidate) = candidates.pop() {
                    // 履歴に追加（上限に達したら古いものを削除）
                    if visited.len() >= self.max_history {
                        visited.pop_front();
                    }
                    visited.push_back(format!("{:?}", candidate.ast));
                    beam.push(candidate);
                } else {
                    break;
                }
            }

            // If beam is empty, we're stuck
            if beam.is_empty() {
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_iterations as u64);
                    pb.finish_with_message(format!("cost {:.2} (beam empty)", initial_cost));
                    pb.tick()
                }
                break;
            }

            if let Some(ref pb) = pb {
                let best_cost = beam
                    .iter()
                    .map(|c| c.cost)
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(initial_cost);
                pb.set_message(format!("beam {}, cost {:.2}", beam.len(), best_cost));
                pb.inc(1);
                pb.tick();
            }
        }

        let best = beam
            .into_iter()
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal))
            .map(|c| c.ast)
            .unwrap_or_else(|| ast.clone());

        if let Some(ref pb) = pb {
            let final_cost = self.estimator.estimate_cost(&best);
            pb.finish_with_message(format!("cost {:.2}", final_cost));
            pb.tick()
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_pattern;
    use crate::opt::ast::heuristic::{
        NodeCountCostEstimator, OperationCostEstimator, RuleBasedSuggester,
    };

    fn i(val: isize) -> AstNode {
        AstNode::from(val)
    }

    #[test]
    fn test_cost_based_optimizer() {
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
    fn test_beam_search_optimizer_multiple_paths() {
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
