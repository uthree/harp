//! 高レベル演算（hlops）のデモ
//!
//! autograd::Tensorの高レベル演算は、基本演算を組み合わせて実装されています。
//! これにより、計算グラフが大きくなりますが、後段の最適化機能で効率化されます。

use harp::autograd::Tensor;
use harp::graph::{DType, Graph};

fn main() {
    env_logger::init();

    println!("=== High-Level Operations (hlops) Demo ===\n");

    // 1. 数学関数（他の演算の組み合わせで実現）
    println!("1. Mathematical functions composed from basic operations:\n");

    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([5])
            .build(),
        true,
    );

    // log(x) = log2(x) / log2(e)
    println!("  log(x):");
    println!("    Implemented as: log2(x) * (1 / log2(e))");
    let log_x = x.log();
    println!("    Graph created (will be optimized later)");
    println!();

    // exp(x) = 2^(x * log2(e))
    println!("  exp(x):");
    println!("    Implemented as: exp2(x * log2(e))");
    let exp_x = x.exp();
    println!("    Graph created (will be optimized later)");
    println!();

    // cos(x) = sin(x + π/2)
    println!("  cos(x):");
    println!("    Implemented as: sin(x + π/2)");
    let cos_x = x.cos();
    println!("    Graph created (will be optimized later)");
    println!();

    // rsqrt(x) = recip(sqrt(x))
    println!("  rsqrt(x):");
    println!("    Implemented as: recip(sqrt(x))");
    let rsqrt_x = x.rsqrt();
    println!("    Graph created (will be optimized later)");
    println!();

    // 2. 代数演算
    println!("2. Algebraic operations:\n");

    // square(x) = x * x
    println!("  square(x):");
    println!("    Implemented as: x * x");
    let squared = x.square();
    println!("    Graph created");
    println!();

    // powi(x, 3) = x * x * x
    println!("  powi(x, 3):");
    println!("    Implemented as: x * x * x");
    let cubed = x.powi(3);
    println!("    Graph created");
    println!();

    // min(a, b) = -max(-a, -b)
    println!("  min(a, b):");
    println!("    Implemented as: -max(-a, -b)");
    let y = Tensor::from_graph_node(
        graph
            .input("y")
            .with_dtype(DType::F32)
            .with_shape([5])
            .build(),
        true,
    );
    let min_xy = x.min(&y);
    println!("    Graph created");
    println!();

    // clamp(x, min, max) = max(x, min).min(max)
    println!("  clamp(x, min_val, max_val):");
    println!("    Implemented as: max(x, min_val).min(max_val)");
    let min_val = Tensor::zeros(vec![5]);
    let max_val = Tensor::ones(vec![5]);
    let clamped = x.clamp(&min_val, &max_val);
    println!("    Graph created");
    println!();

    // 3. 統計演算
    println!("3. Statistical operations:\n");

    let mut graph2 = Graph::new();
    let matrix = Tensor::from_graph_node(
        graph2
            .input("matrix")
            .with_dtype(DType::F32)
            .with_shape([3, 4])
            .build(),
        true,
    );

    // mean(x, axis) = sum(x, axis) / size(axis)
    println!("  mean(x, axis):");
    println!("    Implemented as: sum(x, axis) / size(axis)");
    let mean_val = matrix.mean(1);
    println!("    Graph created");
    println!();

    // variance(x, axis) = mean((x - mean(x))^2, axis)
    println!("  variance(x, axis):");
    println!("    Implemented as: mean((x - mean(x))^2, axis)");
    let var_val = matrix.variance(1);
    println!("    Graph created (includes multiple operations)");
    println!();

    // 4. 設計思想
    println!("4. Design Philosophy:\n");
    println!("  ✓ Simplicity: High-level operations are easy to understand");
    println!("  ✓ Composability: Built from basic operations");
    println!("  ✓ Optimization: Graph optimizer will fuse and simplify");
    println!("  ✓ Maintenance: Easy to add new operations");
    println!();

    println!("Example: cos(x) generates this graph:");
    println!("  1. x + π/2 (Add operation)");
    println!("  2. sin(...) (Sin operation)");
    println!("  → Later optimized into a single 'cos' kernel if beneficial");
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Key points:");
    println!("  - High-level ops are in src/autograd/hlops.rs");
    println!("  - Basic ops remain in src/autograd/tensor.rs");
    println!("  - Graph optimizer handles efficiency");
    println!("  - Same design as graph::hlops module");
}
