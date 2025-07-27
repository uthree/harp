use harp::backends::clang::compiler::ClangCompileOptions;
use harp::optimization::autotuner::{
    Autotuner, BackendOptions, GridSearch, OptimizationRule, SearchSpace,
};
use harp::prelude::*;
use harp::uop::Number;

fn main() {
    // 1. Define the search space for the autotuner.
    let search_space = SearchSpace {
        // Only include rules that are truly optional/tunable.
        tunable_rules: vec![OptimizationRule::RecipRecip],
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
    let a = Tensor::ones(vec![100], DType::F32);
    let b = Tensor::new(
        TensorOp::Constant(Number::F32(2.0)),
        vec![],
        ShapeTracker::new(vec![100]),
        DType::F32,
        backend("clang"),
    );
    let c = Tensor::zeros(vec![100], DType::F32);

    // An expression that can be optimized, e.g., (a * 1.0) + 0.0
    let expr = (a * b) + c;

    println!("Starting autotuning for a simple tensor expression...");

    // 5. Run the autotuner.
    let best_result = tuner.run(&expr);

    // 6. Print the best result.
    println!("\n--- Autotuning Finished ---");
    println!("Best configuration found!");
    println!("  Execution time: {:?}", best_result.execution_time);
    println!("  Enabled rules: {:?}", best_result.config.enabled_rules);
    println!(
        "  Backend options: {:?}",
        best_result.config.backend_options
    );
}
