use crate::ast::AstNode;
use crate::opt::AstOptimizer;
use crate::opt::ast::heuristic::{CostEstimator, Rewrite, RewriteSuggester};
use console::Style;
use indicatif::HumanDuration;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashSet, VecDeque};
use std::time::Instant;

/// An optimizer that uses beam search to find a low-cost AST.
pub struct BeamSearchAstOptimizer<S: RewriteSuggester, C: CostEstimator> {
    suggester: S,
    cost_estimator: C,
    pub beam_width: usize,
    pub max_steps: usize,
    pub max_visited_size: usize,
}

impl<S: RewriteSuggester, C: CostEstimator> BeamSearchAstOptimizer<S, C> {
    pub fn new(suggester: S, cost_estimator: C) -> Self {
        Self {
            suggester,
            cost_estimator,
            beam_width: 4,
            max_steps: 300,
            max_visited_size: 10000, // Default max size for visited cache
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

    pub fn with_max_visited_size(mut self, max_visited_size: usize) -> Self {
        self.max_visited_size = max_visited_size;
        self
    }
}

impl<S: RewriteSuggester, C: CostEstimator> AstOptimizer for BeamSearchAstOptimizer<S, C> {
    fn optimize(&mut self, initial_node: &AstNode) -> AstNode {
        // start time
        let start = Instant::now();

        // Progress bar setup
        let pb = ProgressBar::new(self.max_steps as u64);
        pb.set_style(
            ProgressStyle::with_template("{prefix:>12.cyan.bold} [{bar:24}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_prefix("Optimizing");

        // Initialize beam with the initial node
        let mut beam = vec![(
            initial_node.clone(),
            self.cost_estimator.estimate_cost(initial_node),
        )];
        let mut visited_set = HashSet::new();
        let mut visited_queue = VecDeque::with_capacity(self.max_visited_size);
        visited_set.insert(initial_node.clone());
        visited_queue.push_back(initial_node.clone());

        for i in 0..self.max_steps {
            let mut next_candidates = HashSet::new();
            for (node, _) in &beam {
                let suggestions = self.suggester.suggest(node);
                for suggestion in suggestions {
                    // Replace the original node with the suggestion in the full AST
                    let new_full_node =
                        replace_node_in_ast(node, &suggestion.original, &suggestion.new);
                    if !visited_set.contains(&new_full_node) {
                        next_candidates.insert(new_full_node);
                    }
                }
            }

            let mut candidates = Vec::with_capacity(next_candidates.len() + beam.len());
            for suggestion in next_candidates {
                if visited_queue.len() >= self.max_visited_size
                    && let Some(oldest) = visited_queue.pop_front()
                {
                    visited_set.remove(&oldest);
                }
                let cost = self.cost_estimator.estimate_cost(&suggestion);
                candidates.push((suggestion.clone(), cost));
                visited_set.insert(suggestion.clone());
                visited_queue.push_back(suggestion);
            }

            // Add current beam to candidates to ensure we don't lose good solutions
            candidates.extend(beam.clone());

            // Sort candidates by cost and remove duplicates
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            candidates.dedup_by(|a, b| a.0 == b.0);

            let new_beam = candidates
                .into_iter()
                .take(self.beam_width)
                .collect::<Vec<_>>();

            if new_beam.is_empty() || new_beam == beam {
                beam = new_beam;
                break;
            }
            beam = new_beam;

            pb.tick();
            pb.inc(1);
        }

        pb.finish_and_clear();
        pb.tick();
        let green_bold = Style::new().green().bold();
        pb.println(format!(
            "{:>12} optimize AST with beam search algorithm in {}",
            green_bold.apply_to("Finished"),
            HumanDuration(start.elapsed())
        ));
        pb.tick();

        // Return the best node found
        beam.into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(node, _)| node)
            .unwrap_or_else(|| initial_node.clone())
    }
}

/// Replaces occurrences of `original` with `new` in the `root` AST.
fn replace_node_in_ast(root: &AstNode, original: &AstNode, new: &AstNode) -> AstNode {
    if root == original {
        return new.clone();
    }

    let new_srcs = root
        .src
        .iter()
        .map(|child| replace_node_in_ast(child, original, new))
        .collect();

    AstNode::_new(root.op.clone(), new_srcs, root.dtype.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, AstOp};
    use crate::opt::ast::heuristic::{CostEstimator, RewriteSuggester};
    use std::collections::HashMap;

    // A mock suggester that provides predefined rewrites.
    struct MockRewriteSuggester {
        rules: HashMap<AstNode, Vec<Rewrite>>,
    }

    impl RewriteSuggester for MockRewriteSuggester {
        fn suggest(&self, node: &AstNode) -> Vec<Rewrite> {
            if let Some(rewrites) = self.rules.get(node) {
                return rewrites.clone();
            }
            // Also check children
            let mut suggestions = vec![];
            for child in &node.src {
                suggestions.extend(self.suggest(child));
            }
            suggestions
        }
    }

    // A mock cost estimator that uses the integer value of the node as its cost.
    struct MockCostEstimator;

    impl CostEstimator for MockCostEstimator {
        fn estimate_cost(&self, ast: &AstNode) -> f32 {
            if let AstOp::Const(c) = &ast.op {
                if let Some(val) = c.as_isize() {
                    return val as f32;
                }
            }
            f32::MAX
        }
    }

    #[test]
    fn test_beam_search_optimizer() {
        // Define the rewrite rules for the suggester.
        let mut rules = HashMap::new();
        let node_10 = AstNode::from(10isize);
        let node_8 = AstNode::from(8isize);
        let node_9 = AstNode::from(9isize);
        let node_5 = AstNode::from(5isize);
        rules.insert(
            node_10.clone(),
            vec![
                Rewrite {
                    original: node_10.clone(),
                    new: node_8.clone(),
                },
                Rewrite {
                    original: node_10.clone(),
                    new: node_9.clone(),
                },
            ],
        );
        rules.insert(
            node_8.clone(),
            vec![
                Rewrite {
                    original: node_8.clone(),
                    new: node_5.clone(),
                },
                Rewrite {
                    original: node_8.clone(),
                    new: AstNode::from(6isize),
                },
            ],
        );
        rules.insert(
            node_9.clone(),
            vec![
                Rewrite {
                    original: node_9.clone(),
                    new: AstNode::from(7isize),
                },
                Rewrite {
                    original: node_9.clone(),
                    new: AstNode::from(1isize),
                },
            ],
        );
        rules.insert(
            node_5.clone(),
            vec![Rewrite {
                original: node_5.clone(),
                new: AstNode::from(2isize),
            }],
        );

        let suggester = MockRewriteSuggester { rules };
        let cost_estimator = MockCostEstimator;

        // Create the optimizer.
        let mut optimizer = BeamSearchAstOptimizer::new(suggester, cost_estimator)
            .with_beam_width(2)
            .with_max_steps(3);

        // The initial node to optimize.
        let initial_node = node_10;

        // Run the optimization.
        let optimized_node = optimizer.optimize(&initial_node);

        // The expected result should be the node with the lowest cost (1).
        let expected_node = AstNode::from(1isize);

        assert_eq!(optimized_node, expected_node);
    }

    #[test]
    fn test_optimizer_returns_initial_if_no_better_found() {
        let suggester = MockRewriteSuggester {
            rules: HashMap::new(),
        };
        let cost_estimator = MockCostEstimator;
        let mut optimizer = BeamSearchAstOptimizer::new(suggester, cost_estimator)
            .with_beam_width(2)
            .with_max_steps(3);
        let initial_node = AstNode::from(10isize);
        let optimized_node = optimizer.optimize(&initial_node);
        assert_eq!(optimized_node, initial_node);
    }
}
