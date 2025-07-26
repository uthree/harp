use harp::autotuner::{Autotuner, BackendOptions, GridSearch, OptimizationRule, SearchSpace};
use harp::backends::c::compiler::ClangCompileOptions;
use harp::prelude::*;

fn main() {
    // 1. Define the search space for the autotuner.
    let search_space = SearchSpace {
        tunable_rules: vec![
            OptimizationRule::AddZero,
            OptimizationRule::MulOne,
            OptimizationRule::MulZero,
            OptimizationRule::RecipRecip,
        ],
        // Define a list of backend options to try.
        tunable_backend_options: {
            let mut options = Vec::new();
            for &opt_level in &[0, 1, 2, 3] {
                for &fast_math in &[true, false] {
                    options.push(BackendOptions::Clang(ClangCompileOptions {
                        optimization_level: opt_level,
                        use_fast_math: fast_math,
                        debug_info: false,
                    }));
                }
            }
            options
        },
    };

    // 2. Choose a search strategy.
    let strategy = GridSearch::new(&search_space);

    // 3. Initialize the autotuner.
    let mut tuner = Autotuner::new(&search_space, strategy);

    // 4. Create the tensors for the computation we want to tune.
    let a = Tensor::from_vec(vec![1.0f32; 100], &[100]);
    let b = Tensor::from_vec(vec![2.0f32; 100], &[100]);
    let c = Tensor::from_vec(vec![0.0f32; 100], &[100]);

    // An expression that can be optimized, e.g., (a * 1.0) + 0.0
    let expr = (a * b) + c;

    println!("Starting autotuning for a simple tensor expression...");

    // 5. Run the autotuner with a limit of 16 trials.
    tuner.run(&expr, Some(16));

    // 6. Print the best result.
    if let Some(best) = tuner.best_result() {
        println!("\n--- Autotuning Finished ---");
        println!("Best configuration found!");
        println!("  Execution time: {:?}", best.execution_time);
        println!("  Enabled rules: {:?}", best.config.enabled_rules);
        println!("  Backend options: {:?}", best.config.backend_options);
    } else {
        println!("No successful configuration found.");
    }
}
