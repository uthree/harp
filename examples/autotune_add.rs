use harp::autotuner::{Autotuner, GridSearch, OptimizationRule, SearchSpace};
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
        optimization_levels: vec![0, 1, 2, 3],
        use_fast_math: vec![true, false],
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
        println!(
            "  Clang options: -O{} {}",
            best.config.clang_options.optimization_level,
            if best.config.clang_options.use_fast_math {
                "-ffast-math"
            } else {
                ""
            }
        );
    } else {
        println!("No successful configuration found.");
    }
}
