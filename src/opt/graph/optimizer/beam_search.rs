//! Beam search optimizer for computation graphs
//!
//! Similar to the AST beam search optimizer, but operates on the computation graph
//! before lowering. This enables high-level pattern detection and optimization.

use std::cmp::Ordering;

use log::{debug, info, trace};

use super::super::{GraphCostEstimator, GraphOptimizer, GraphSuggestResult, GraphSuggester};
use crate::graph::GraphNode;
use crate::opt::graph::estimator::SimpleGraphCostEstimator;
use crate::opt::progress::{IndicatifProgress, NoOpProgress, ProgressState, SearchProgress};

/// A snapshot of the graph optimization state at a specific step
#[derive(Clone)]
pub struct GraphOptimizationSnapshot {
    /// Step number
    pub step: usize,
    /// Current graph roots
    pub roots: Vec<GraphNode>,
    /// Estimated cost at this step
    pub cost: f32,
    /// Description of the optimization applied
    pub description: String,
    /// Name of the suggester that proposed this change
    pub suggester_name: Option<String>,
    /// Alternative candidates that were not selected
    pub alternatives: Vec<GraphAlternativeCandidate>,
}

/// An alternative candidate that was not selected
#[derive(Clone)]
pub struct GraphAlternativeCandidate {
    /// Alternative graph roots
    pub roots: Vec<GraphNode>,
    /// Estimated cost for this alternative
    pub cost: f32,
    /// Name of the suggester
    pub suggester_name: Option<String>,
    /// Description
    pub description: String,
}

/// History of graph optimization steps
#[derive(Default)]
pub struct GraphOptimizationHistory {
    snapshots: Vec<GraphOptimizationSnapshot>,
    /// Target backend for this optimization
    target_backend: crate::backend::TargetBackend,
}

impl GraphOptimizationHistory {
    /// Create an empty history
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            target_backend: crate::backend::TargetBackend::Generic,
        }
    }

    /// Create a new history with a specific target backend
    pub fn with_target_backend(target_backend: crate::backend::TargetBackend) -> Self {
        Self {
            snapshots: Vec::new(),
            target_backend,
        }
    }

    /// Get the target backend
    pub fn target_backend(&self) -> crate::backend::TargetBackend {
        self.target_backend
    }

    /// Set the target backend
    pub fn set_target_backend(&mut self, backend: crate::backend::TargetBackend) {
        self.target_backend = backend;
    }

    /// Add a snapshot to the history
    pub fn push(&mut self, snapshot: GraphOptimizationSnapshot) {
        self.snapshots.push(snapshot);
    }

    /// Get the number of snapshots
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if the history is empty
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Get a snapshot by index
    pub fn get(&self, index: usize) -> Option<&GraphOptimizationSnapshot> {
        self.snapshots.get(index)
    }

    /// Get an iterator over snapshots
    pub fn iter(&self) -> impl Iterator<Item = &GraphOptimizationSnapshot> {
        self.snapshots.iter()
    }

    /// Consume and return the snapshots
    pub fn into_snapshots(self) -> Vec<GraphOptimizationSnapshot> {
        self.snapshots
    }
}

/// A candidate in the beam with its optimization path
#[derive(Clone)]
struct BeamEntry {
    /// The graph roots for this candidate
    roots: Vec<GraphNode>,
    /// The path of transformations applied to reach this state
    path: Vec<(String, String)>, // (suggester_name, description)
}

/// Beam search optimizer for computation graphs
///
/// This optimizer explores the space of possible graph transformations
/// using beam search, keeping the top `beam_width` candidates at each step.
///
/// # Termination
///
/// The optimization terminates when:
/// - Maximum steps reached
/// - No new suggestions from any suggester
/// - No improvement for `max_no_improvement_steps` consecutive steps
pub struct GraphBeamSearchOptimizer<S, E = SimpleGraphCostEstimator, P = IndicatifProgress>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
    P: SearchProgress,
{
    suggester: S,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    progress: Option<P>,
    /// Steps without improvement before early termination
    max_no_improvement_steps: Option<usize>,
    /// Whether to record optimization history
    record_history: bool,
    /// Recorded optimization history
    history: Option<GraphOptimizationHistory>,
    /// Target backend for optimization
    target_backend: crate::backend::TargetBackend,
}

impl<S> GraphBeamSearchOptimizer<S, SimpleGraphCostEstimator, IndicatifProgress>
where
    S: GraphSuggester,
{
    /// Create a new beam search optimizer with default settings
    pub fn new(suggester: S) -> Self {
        Self {
            suggester,
            estimator: SimpleGraphCostEstimator::new(),
            beam_width: 5, // Smaller than AST optimizer since graph has fewer transformations
            max_steps: 100, // Also smaller
            progress: if cfg!(debug_assertions) {
                Some(IndicatifProgress::new())
            } else {
                None
            },
            max_no_improvement_steps: Some(3),
            record_history: false,
            history: None,
            target_backend: crate::backend::TargetBackend::Generic,
        }
    }
}

impl<S, E, P> GraphBeamSearchOptimizer<S, E, P>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
    P: SearchProgress,
{
    /// Set a custom cost estimator
    pub fn with_estimator<NewE>(self, estimator: NewE) -> GraphBeamSearchOptimizer<S, NewE, P>
    where
        NewE: GraphCostEstimator,
    {
        GraphBeamSearchOptimizer {
            suggester: self.suggester,
            estimator,
            beam_width: self.beam_width,
            max_steps: self.max_steps,
            progress: self.progress,
            max_no_improvement_steps: self.max_no_improvement_steps,
            record_history: self.record_history,
            history: self.history,
            target_backend: self.target_backend,
        }
    }

    /// Set beam width
    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    /// Set maximum steps
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// Set a custom progress reporter
    pub fn with_progress<P2: SearchProgress>(
        self,
        progress: P2,
    ) -> GraphBeamSearchOptimizer<S, E, P2> {
        GraphBeamSearchOptimizer {
            suggester: self.suggester,
            estimator: self.estimator,
            beam_width: self.beam_width,
            max_steps: self.max_steps,
            progress: Some(progress),
            max_no_improvement_steps: self.max_no_improvement_steps,
            record_history: self.record_history,
            history: self.history,
            target_backend: self.target_backend,
        }
    }

    /// Disable progress reporting
    pub fn without_progress(self) -> GraphBeamSearchOptimizer<S, E, NoOpProgress> {
        GraphBeamSearchOptimizer {
            suggester: self.suggester,
            estimator: self.estimator,
            beam_width: self.beam_width,
            max_steps: self.max_steps,
            progress: None,
            max_no_improvement_steps: self.max_no_improvement_steps,
            record_history: self.record_history,
            history: self.history,
            target_backend: self.target_backend,
        }
    }

    /// Enable history recording for visualization
    pub fn with_history(mut self) -> Self {
        self.record_history = true;
        self.history = Some(GraphOptimizationHistory::with_target_backend(self.target_backend));
        self
    }

    /// Set the target backend for optimization
    ///
    /// Used by visualization tools to auto-select the appropriate renderer.
    pub fn with_target_backend(mut self, backend: crate::backend::TargetBackend) -> Self {
        self.target_backend = backend;
        // Update history if it exists
        if let Some(ref mut history) = self.history {
            history.set_target_backend(backend);
        }
        self
    }

    /// Take the recorded optimization history
    ///
    /// Returns None if history recording was not enabled or history was already taken.
    pub fn take_history(&mut self) -> Option<GraphOptimizationHistory> {
        self.history.take()
    }

    /// Set early termination threshold
    pub fn with_no_improvement_limit(mut self, steps: Option<usize>) -> Self {
        self.max_no_improvement_steps = steps;
        self
    }

    /// Select top N candidates by cost
    fn select_top_n(
        &self,
        candidates: Vec<(Vec<GraphNode>, String, String, Vec<(String, String)>)>,
        n: usize,
    ) -> Vec<(Vec<GraphNode>, f32, String, String, Vec<(String, String)>)> {
        let mut with_cost: Vec<_> = candidates
            .into_iter()
            .map(|(roots, name, desc, path)| {
                let cost = self.estimator.estimate(&roots);
                (roots, cost, name, desc, path)
            })
            .collect();

        with_cost.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        with_cost.into_iter().take(n).collect()
    }
}

impl<S, E, P> GraphOptimizer for GraphBeamSearchOptimizer<S, E, P>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
    P: SearchProgress,
{
    fn optimize(&mut self, roots: Vec<GraphNode>) -> Vec<GraphNode> {
        info!(
            "Graph beam search optimization started (beam_width={}, max_steps={})",
            self.beam_width, self.max_steps
        );

        let mut beam = vec![BeamEntry {
            roots: roots.clone(),
            path: vec![],
        }];

        let initial_cost = self.estimator.estimate(&roots);
        info!("Initial graph cost: {:.2e}", initial_cost);

        // Record initial state if history is enabled
        if let Some(ref mut history) = self.history {
            history.push(GraphOptimizationSnapshot {
                step: 0,
                roots: roots.clone(),
                cost: initial_cost,
                description: "Initial graph".to_string(),
                suggester_name: None,
                alternatives: Vec::new(),
            });
        }

        let mut global_best = BeamEntry {
            roots,
            path: vec![],
        };
        let mut global_best_cost = initial_cost;

        if let Some(ref mut progress) = self.progress {
            progress.start(self.max_steps, "Graph optimization");
        }

        let mut no_improvement_count = 0;
        let mut best_cost = initial_cost;

        for step in 0..self.max_steps {
            if let Some(ref mut progress) = self.progress {
                progress.update(&ProgressState::new(
                    step,
                    self.max_steps,
                    format!("step {}", step + 1),
                ));
            }

            // Generate candidates from all beam entries
            let mut candidates: Vec<(Vec<GraphNode>, String, String, Vec<(String, String)>)> =
                Vec::new();

            for entry in &beam {
                let suggestions = self.suggester.suggest(&entry.roots);
                for GraphSuggestResult {
                    roots,
                    suggester_name,
                    description,
                } in suggestions
                {
                    let mut new_path = entry.path.clone();
                    new_path.push((suggester_name.clone(), description.clone()));
                    candidates.push((roots, suggester_name, description, new_path));
                }
            }

            if candidates.is_empty() {
                info!(
                    "No more candidates at step {} - optimization complete",
                    step
                );
                break;
            }

            let num_candidates = candidates.len();
            trace!("Found {} candidates at step {}", num_candidates, step);

            // Select top candidates
            let selected = self.select_top_n(candidates, self.beam_width);

            // Update beam
            beam = selected
                .iter()
                .map(|(roots, _cost, _name, _desc, path)| BeamEntry {
                    roots: roots.clone(),
                    path: path.clone(),
                })
                .collect();

            // Check best candidate and record history
            if let Some((roots, cost, name, desc, _path)) = selected.first() {
                // Record this step in history if enabled
                if let Some(ref mut history) = self.history {
                    let alternatives: Vec<GraphAlternativeCandidate> = selected
                        .iter()
                        .skip(1)
                        .map(|(alt_roots, alt_cost, alt_name, alt_desc, _)| {
                            GraphAlternativeCandidate {
                                roots: alt_roots.clone(),
                                cost: *alt_cost,
                                suggester_name: Some(alt_name.clone()),
                                description: alt_desc.clone(),
                            }
                        })
                        .collect();

                    history.push(GraphOptimizationSnapshot {
                        step: step + 1,
                        roots: roots.clone(),
                        cost: *cost,
                        description: desc.clone(),
                        suggester_name: Some(name.clone()),
                        alternatives,
                    });
                }

                if *cost >= best_cost {
                    no_improvement_count += 1;

                    if let Some(max_no_improvement) = self.max_no_improvement_steps {
                        debug!(
                            "Step {}: no improvement (current={:.2e}, best={:.2e}, {}/{})",
                            step, cost, best_cost, no_improvement_count, max_no_improvement
                        );

                        if no_improvement_count >= max_no_improvement {
                            info!(
                                "No cost improvement for {} steps - optimization complete",
                                max_no_improvement
                            );
                            break;
                        }
                    }
                } else {
                    no_improvement_count = 0;
                    let improvement_pct = (best_cost - cost) / best_cost * 100.0;
                    info!(
                        "Step {}: cost improved {:.2e} -> {:.2e} ({:+.1}%) via {} - {}",
                        step, best_cost, cost, -improvement_pct, name, desc
                    );
                    best_cost = *cost;
                    global_best = BeamEntry {
                        roots: roots.clone(),
                        path: vec![],
                    };
                    global_best_cost = *cost;
                }
            }
        }

        if let Some(ref mut progress) = self.progress {
            progress.finish(&crate::opt::progress::FinishInfo::new(
                std::time::Duration::ZERO,
                self.max_steps,
                self.max_steps,
                "Graph optimization",
            ));
        }

        let improvement_pct = if initial_cost > 0.0 {
            (initial_cost - global_best_cost) / initial_cost * 100.0
        } else {
            0.0
        };
        info!(
            "Graph optimization complete: cost {:.2e} -> {:.2e} ({:+.1}%)",
            initial_cost, global_best_cost, -improvement_pct
        );

        if !global_best.path.is_empty() {
            debug!(
                "Optimization path: {}",
                global_best
                    .path
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            );
        }

        global_best.roots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::{Expr, input};
    use crate::opt::graph::suggesters::CompositeSuggester;

    struct DummySuggester;

    impl GraphSuggester for DummySuggester {
        fn name(&self) -> &str {
            "dummy"
        }

        fn suggest(&self, _roots: &[GraphNode]) -> Vec<GraphSuggestResult> {
            vec![]
        }
    }

    #[test]
    fn test_optimizer_no_suggestions() {
        let suggester = DummySuggester;
        let mut optimizer = GraphBeamSearchOptimizer::new(suggester)
            .without_progress()
            .with_max_steps(10);

        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = &x + &x;

        let result = optimizer.optimize(vec![y.clone()]);

        // With no suggestions, should return the original
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_optimizer_empty_graph() {
        let suggester = CompositeSuggester::new(vec![]);
        let mut optimizer = GraphBeamSearchOptimizer::new(suggester).without_progress();

        let result = optimizer.optimize(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_optimizer_parameters() {
        let suggester = DummySuggester;
        let optimizer = GraphBeamSearchOptimizer::new(suggester)
            .with_beam_width(10)
            .with_max_steps(50)
            .with_no_improvement_limit(Some(5))
            .without_progress();

        assert_eq!(optimizer.beam_width, 10);
        assert_eq!(optimizer.max_steps, 50);
        assert_eq!(optimizer.max_no_improvement_steps, Some(5));
    }
}
