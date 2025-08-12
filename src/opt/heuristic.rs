use crate::ast::{AstNode, AstOp};
use crate::backend::Renderer;
use crate::backend::c::renderer::CRenderer;
use crate::opt::ast::{
    CostEstimator, DeterministicAstOptimizer, OptimizationSuggester, RewriteRule,
};
use console::Style;
use indicatif::HumanDuration;
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use rustc_hash::FxHashSet;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::rc::Rc;
use std::time::Instant;

/// A suggester that uses a set of rewrite rules to generate optimization candidates.
#[derive(Clone)]
pub struct RuleBasedSuggester {
    rules: Vec<Rc<RewriteRule>>,
}

impl RuleBasedSuggester {
    pub fn new(rules: Vec<Rc<RewriteRule>>) -> Self {
        Self { rules }
    }
}

impl OptimizationSuggester for RuleBasedSuggester {
    fn suggest_optimizations(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();
        for rule in &self.rules {
            if let Some(captures) = rule.capture(node) {
                let rewritten_node = (rule.rewriter)(captures);
                if &rewritten_node != node {
                    suggestions.push(rewritten_node);
                }
            }
        }
        suggestions
    }
}

/// A suggester that combines multiple suggesters into one.
pub struct CompositeSuggester {
    suggesters: Vec<Box<dyn OptimizationSuggester>>,
}

impl CompositeSuggester {
    pub fn new(suggesters: Vec<Box<dyn OptimizationSuggester>>) -> Self {
        Self { suggesters }
    }
}

impl OptimizationSuggester for CompositeSuggester {
    fn suggest_optimizations(&self, node: &AstNode) -> Vec<AstNode> {
        self.suggesters
            .iter()
            .flat_map(|s| s.suggest_optimizations(node))
            .collect()
    }
}

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

#[derive(Clone, Copy)]
pub struct HandcodedCostEstimator;

impl CostEstimator for HandcodedCostEstimator {
    fn estimate_cost(&self, node: &AstNode) -> f32 {
        let mut count = 0.0;
        count += match &node.op {
            AstOp::Recip => 5.0,
            AstOp::Store => 100.0,
            AstOp::Range { loop_var: _ } => 200.0,
            AstOp::Assign => 100.0,
            AstOp::Declare {
                name: _name,
                dtype: _dtype,
            } => 100.0,
            AstOp::Func {
                name: _name,
                args: _args,
            } => 1.0,
            AstOp::Deref => 50.0,
            AstOp::Var(_) => 10.0,
            AstOp::Call(_) => 20.0,
            _ => 1.0,
        };
        for child in &node.src {
            count += self.estimate_cost(child);
        }
        count
    }
}

/// An optimizer that uses a greedy search to find a low-cost AST.
pub struct GreedyAstOptimizer<S: OptimizationSuggester, C: CostEstimator> {
    suggester: S,
    cost_estimator: C,
    pub max_iterations: usize,
}

impl<S: OptimizationSuggester, C: CostEstimator> GreedyAstOptimizer<S, C> {
    pub fn new(suggester: S, cost_estimator: C) -> Self {
        Self {
            suggester,
            cost_estimator,
            max_iterations: 1000, // Default max iterations
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
    for GreedyAstOptimizer<S, C>
{
    fn optimize(&self, mut node: AstNode) -> AstNode {
        let start = Instant::now();

        // Create a progress bar to visualize the optimization process.
        let pb = ProgressBar::new(self.max_iterations as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{prefix:>12.cyan.bold} [{bar:57}] {pos}/{len} {wide_msg}",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        pb.set_prefix("Optimizing");
        pb.println("Starting greedy optimization...");

        for i in 0..self.max_iterations {
            let original_node = node.clone();
            let new_node = self.apply_one_pass(original_node.clone());

            if new_node == original_node {
                debug!("Greedy search reached fixed point after {i} iterations.");
                pb.finish_and_clear();

                // compilation is finished
                let green_bold = Style::new().green().bold();
                println!(
                    "{:>12} optimize AST with greedy search algorithm in {}",
                    green_bold.apply_to("Finished"),
                    HumanDuration(start.elapsed())
                );
                return new_node;
            }
            debug!("AST changed in iteration {i}. Continuing greedy search...");
            let cost = self.cost_estimator.estimate_cost(&new_node);
            pb.set_message(format!("Cost: {cost:.2}"));
            let mut renderer = CRenderer::new();
            let code = renderer.render(new_node.clone());
            pb.println(format!("Iteration {i}: Cost {cost:.2}\n---\n{code}\n---"));
            pb.inc(1);
            node = new_node;
        }
        debug!(
            "Greedy search finished after reaching max iterations ({})",
            self.max_iterations
        );
        pb.finish_and_clear();

        // compilation is finished
        let green_bold = Style::new().green().bold();
        println!(
            "{:>12} optimize AST with greedy search algorithm in {}",
            green_bold.apply_to("Finished"),
            HumanDuration(start.elapsed())
        );
        node
    }
}

// Helper struct to use AstNode in a min-heap (BinaryHeap)
#[derive(Clone)]
struct CostAstNode(f32, AstNode);

impl PartialEq for CostAstNode {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for CostAstNode {}

// We want a min-heap, so we implement Ord to reverse the comparison
impl Ord for CostAstNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for CostAstNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// An optimizer that uses beam search to find a low-cost AST.
pub struct BeamSearchAstOptimizer<S: OptimizationSuggester, C: CostEstimator> {
    suggester: S,
    cost_estimator: C,
    pub beam_width: usize,
    pub max_steps: usize,
}

impl<S: OptimizationSuggester, C: CostEstimator> BeamSearchAstOptimizer<S, C> {
    pub fn new(suggester: S, cost_estimator: C, beam_width: usize) -> Self {
        Self {
            suggester,
            cost_estimator,
            beam_width,
            max_steps: 1000, // Default max steps for the search
        }
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Recursively finds all possible single-mutation variants of a given AST.
    fn find_all_single_mutations(&self, node: &AstNode) -> Vec<AstNode> {
        let mut all_mutations = Vec::new();

        // 1. Mutations for the current node (top-level)
        let top_level_suggestions = self.suggester.suggest_optimizations(node);
        all_mutations.extend(top_level_suggestions);

        // 2. Mutations from children
        for (i, child) in node.src.iter().enumerate() {
            let child_mutations = self.find_all_single_mutations(child);
            for child_mutation in child_mutations {
                let mut new_src = node.src.clone();
                new_src[i] = child_mutation;
                let new_parent = AstNode::new(node.op.clone(), new_src, node.dtype.clone());
                all_mutations.push(new_parent);
            }
        }
        all_mutations
    }
}

impl<S: OptimizationSuggester, C: CostEstimator> DeterministicAstOptimizer
    for BeamSearchAstOptimizer<S, C>
{
    fn optimize(&self, node: AstNode) -> AstNode {
        let mut beam: Vec<AstNode> = vec![node];
        let mut visited: FxHashSet<AstNode> = FxHashSet::from_iter(beam.iter().cloned());

        // Create a progress bar to visualize the optimization process.
        let start = Instant::now();
        let pb = ProgressBar::new(self.max_steps as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{prefix:>12.cyan.bold} [{bar:57}] {pos}/{len} {wide_msg}",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        pb.set_prefix("Optimizing");
        pb.println("Starting beam search...");

        for step in 0..self.max_steps {
            let mut candidates = BinaryHeap::new();

            for ast_in_beam in &beam {
                // Add the current ast to candidates to ensure we don't get worse
                candidates.push(CostAstNode(
                    self.cost_estimator.estimate_cost(ast_in_beam),
                    ast_in_beam.clone(),
                ));

                let all_possible_next_asts = self.find_all_single_mutations(ast_in_beam);

                for next_ast in all_possible_next_asts {
                    if !visited.contains(&next_ast) {
                        let cost = self.cost_estimator.estimate_cost(&next_ast);
                        candidates.push(CostAstNode(cost, next_ast));
                    }
                }
            }

            // Select the top `beam_width` candidates for the new beam
            let mut new_beam = Vec::with_capacity(self.beam_width);
            let mut new_beam_set = FxHashSet::default();
            while let Some(CostAstNode(_, candidate_node)) = candidates.pop() {
                if new_beam.len() >= self.beam_width {
                    break;
                }
                if new_beam_set.insert(candidate_node.clone()) {
                    new_beam.push(candidate_node);
                }
            }

            let old_beam_set: FxHashSet<_> = beam.iter().cloned().collect();
            if new_beam.is_empty() || new_beam_set == old_beam_set {
                debug!("Beam search reached fixed point after {} steps.", step);
                pb.finish_and_clear();

                // compilation is finished
                let green_bold = Style::new().green().bold();
                println!(
                    "{:>12} optimize AST with beam search algorithm in {}",
                    green_bold.apply_to("Finished"),
                    HumanDuration(start.elapsed())
                );
                break;
            }

            beam = new_beam;
            visited.extend(beam.iter().cloned());
            pb.println(format!(
                "Beam search step {}: beam size = {}",
                step,
                beam.len()
            ));

            let best_node = beam
                .iter()
                .min_by(|a, b| {
                    self.cost_estimator
                        .estimate_cost(a)
                        .partial_cmp(&self.cost_estimator.estimate_cost(b))
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap();
            let cost = self.cost_estimator.estimate_cost(best_node);
            let beam_len = beam.len();
            pb.set_message(format!("Cost: {cost:.2}, Beam: {beam_len}"));
            let mut renderer = CRenderer::new();
            let code = renderer.render(best_node.clone());
            pb.println(format!(
                "Step {step}: Cost {cost:.2}, Beam Size {beam_len}\n---\n{code}\n---"
            ));
            pb.inc(1);
        }
        pb.finish_and_clear();
        let green_bold = Style::new().green().bold();
        println!(
            "{:>12} optimize AST with beam search algorithm in {}",
            green_bold.apply_to("Finished"),
            HumanDuration(start.elapsed())
        );
        // Return the best node found
        beam.into_iter()
            .min_by(|a, b| {
                self.cost_estimator
                    .estimate_cost(a)
                    .partial_cmp(&self.cost_estimator.estimate_cost(b))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;
    use crate::opt::ast::RewriteRule;

    fn setup_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_composite_suggester() {
        setup_logger();
        let node_a = AstNode::var("A");
        let node_b = AstNode::var("B");
        let node_c = AstNode::var("C");

        let rule1 = RewriteRule::new("A->B", node_a.clone(), move |_| node_b.clone());
        let suggester1 = RuleBasedSuggester::new(vec![rule1]);

        let rule2 = RewriteRule::new("A->C", node_a.clone(), move |_| node_c.clone());
        let suggester2 = RuleBasedSuggester::new(vec![rule2]);

        let composite_suggester =
            CompositeSuggester::new(vec![Box::new(suggester1), Box::new(suggester2)]);

        let suggestions = composite_suggester.suggest_optimizations(&node_a);
        assert_eq!(suggestions.len(), 2);
        let node_b_from_suggestion = AstNode::var("B");
        let node_c_from_suggestion = AstNode::var("C");
        assert!(suggestions.contains(&node_b_from_suggestion));
        assert!(suggestions.contains(&node_c_from_suggestion));
    }

    #[test]
    fn test_beam_search_finds_better_solution_than_greedy() {
        setup_logger();
        use crate::ast::AstOp;
        use crate::opt::ast::RewriteRule;

        // A -> B (cost increases), B -> C (cost decreases, lower than A)
        let node_a = AstNode::var("A");
        let node_b = AstNode::var("B");
        let node_c = AstNode::var("C");

        // Dummy rules that match specific AstNodes
        let rule_a_to_b = RewriteRule::new("A->B", AstNode::var("A"), move |_| node_b.clone());
        let rule_b_to_c = RewriteRule::new("B->C", AstNode::var("B"), move |_| node_c.clone());

        let suggester = RuleBasedSuggester::new(vec![rule_a_to_b, rule_b_to_c]);

        #[derive(Clone, Copy)]
        struct CustomEstimator;
        impl CostEstimator for CustomEstimator {
            fn estimate_cost(&self, node: &AstNode) -> f32 {
                if let AstOp::Var(name) = &node.op {
                    return match name.as_str() {
                        "A" => 10.0,
                        "B" => 15.0, // Higher cost intermediate state
                        "C" => 5.0,  // Lowest cost final state
                        _ => 99.0,
                    };
                }
                99.0
            }
        }
        let cost_estimator = CustomEstimator;

        // --- Test with Greedy Search (Beam Width = 1) ---
        let greedy_optimizer =
            BeamSearchAstOptimizer::new(suggester.clone(), cost_estimator, 1).with_max_steps(3);
        let greedy_result = greedy_optimizer.optimize(node_a.clone());
        // Greedy gets stuck at A because the only move is to B, which has a higher cost.
        assert_eq!(greedy_result, node_a);

        // --- Test with Beam Search (Beam Width = 2) ---
        let beam_optimizer =
            BeamSearchAstOptimizer::new(suggester, cost_estimator, 2).with_max_steps(3);
        let beam_result = beam_optimizer.optimize(node_a.clone());
        // Beam search can move to B (cost 15) and keep it in the beam, then find C (cost 5).
        let expected_node_c = AstNode::var("C");
        assert_eq!(beam_result, expected_node_c);
    }
}
