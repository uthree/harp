use crate::backends::c::compiler::ClangCompileOptions;
use crate::dtype::IntoDType;
use crate::tensor::Tensor;
use rustc_hash::FxHashSet;
use std::time::{Duration, Instant};

/// Identifies a specific optimization rule that can be enabled or disabled.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationRule {
    AddZero,
    MulOne,
    MulZero,
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

    pub fn run<T: Clone + Default + 'static + IntoDType>(
        &mut self,
        tensor: &Tensor<T>,
        limit: Option<usize>,
    ) {
        let mut count = 0;
        while let Some(config) = self.strategy.next_config() {
            if let Some(limit) = limit {
                if count >= limit {
                    println!("\nAutotuner reached trial limit of {limit}.");
                    break;
                }
            }
            count += 1;

            println!("Trying config #{count}: {config:?}");
            tensor.clear_cache();

            let start = Instant::now();
            // TODO: Add proper error handling from realize_with_config
            let _ = tensor.realize_with_config(&config);
            let execution_time = start.elapsed();

            let result = TrialResult {
                config,
                execution_time,
                error: None,
            };
            self.results.push(result);
        }
    }

    pub fn best_result(&self) -> Option<&TrialResult> {
        self.results
            .iter()
            .filter(|r| r.error.is_none())
            .min_by_key(|r| r.execution_time)
    }
}
