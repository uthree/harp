use crate::backends::clang::compiler::ClangCompileOptions;
use crate::tensor::Tensor;
use log::debug;
use rustc_hash::FxHashSet;
use std::time::Duration;

/// Identifies a specific optimization rule that can be enabled or disabled by the autotuner.
/// These should be rules that might not be beneficial in all cases.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationRule {
    // Example: (a.recip()).recip() => a.
    // This might not always be faster depending on the backend (e.g., if RECIP is a single fast instruction).
    RecipRecip,
}

/// A generic container for backend-specific compilation options.
/// This allows the autotuner to be backend-agnostic.
#[derive(Debug, Clone)]
pub enum BackendOptions {
    Clang(ClangCompileOptions),
    // Cuda(CudaCompileOptions), // Future extension
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self::Clang(ClangCompileOptions::default())
    }
}

/// A complete set of tunable parameters for a single compilation and execution trial.
#[derive(Debug, Clone, Default)]
pub struct Configuration {
    /// The set of tunable optimization rules to be enabled for this trial.
    pub enabled_rules: FxHashSet<OptimizationRule>,
    pub backend_options: BackendOptions,
}

/// Represents the outcome of a single autotuning trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub config: Configuration,
    pub execution_time: Duration,
    pub error: Option<String>,
}

/// Defines the search space for the autotuner.
#[derive(Debug)]
pub struct SearchSpace {
    pub tunable_rules: Vec<OptimizationRule>,
    pub tunable_backend_options: Vec<BackendOptions>,
}

/// A trait for defining different strategies to explore the `SearchSpace`.
pub trait SearchStrategy {
    fn next_config(&mut self) -> Option<Configuration>;
}

/// A simple search strategy that tries all possible combinations (grid search).
pub struct GridSearch<'a> {
    search_space: &'a SearchSpace,
    // (rules_idx, backend_opts_idx)
    state: (usize, usize),
    num_rule_combinations: usize,
}

impl<'a> GridSearch<'a> {
    pub fn new(search_space: &'a SearchSpace) -> Self {
        Self {
            search_space,
            state: (0, 0),
            num_rule_combinations: 1 << search_space.tunable_rules.len(),
        }
    }
}

impl<'a> SearchStrategy for GridSearch<'a> {
    fn next_config(&mut self) -> Option<Configuration> {
        let (rules_idx, backend_opts_idx) = self.state;

        if backend_opts_idx >= self.search_space.tunable_backend_options.len() {
            return None; // Exhausted
        }

        // --- Generate config for current state ---
        let mut enabled_rules = FxHashSet::default();
        for (i, rule) in self.search_space.tunable_rules.iter().enumerate() {
            if (rules_idx >> i) & 1 == 1 {
                enabled_rules.insert(rule.clone());
            }
        }

        let config = Configuration {
            enabled_rules,
            backend_options: self.search_space.tunable_backend_options[backend_opts_idx].clone(),
        };

        // --- Advance state for next call ---
        let mut next_state = self.state;
        next_state.0 += 1; // rules_idx
        if next_state.0 >= self.num_rule_combinations {
            next_state.0 = 0;
            next_state.1 += 1; // backend_opts_idx
        }
        self.state = next_state;

        Some(config)
    }
}

/// The main autotuner struct.
pub struct Autotuner<'a, S: SearchStrategy> {
    strategy: S,
    results: Vec<TrialResult>,
    _search_space: &'a SearchSpace, // Keep a reference to the search space
}

impl<'a, S: SearchStrategy> Autotuner<'a, S> {
    pub fn new(search_space: &'a SearchSpace, strategy: S) -> Self {
        Self {
            strategy,
            results: Vec::new(),
            _search_space: search_space,
        }
    }

    pub fn run(&mut self, tensor: &Tensor) -> TrialResult {
        let mut best_result: Option<TrialResult> = None;

        while let Some(config) = self.strategy.next_config() {
            debug!("Running trial with config: {config:?}");

            // Clear cache before each run
            tensor.clear_cache();

            let (buffer, duration) = tensor.realize_with_config(&config);
            let result = TrialResult {
                config,
                execution_time: duration,
                error: None,
            };

            if best_result.is_none()
                || result.execution_time < best_result.as_ref().unwrap().execution_time
            {
                best_result = Some(result);
            }
        }
        best_result.unwrap()
    }

    pub fn best_result(&self) -> Option<&TrialResult> {
        self.results
            .iter()
            .filter(|r| r.error.is_none())
            .min_by_key(|r| r.execution_time)
    }
}
