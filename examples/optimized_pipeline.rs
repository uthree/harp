//! 最適化機能を組み込んだGenericPipelineの使用例

use harp::backend::openmp::{CCompiler, CRenderer};
use harp::backend::{AstOptimizationConfig, GenericPipeline, GraphOptimizationConfig, Pipeline};
use harp::graph::{DType, Graph};

fn main() {
    println!("=== GenericPipeline最適化デモ ===\n");

    // 計算グラフを作成
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![100, 200])
        .build();

    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![100, 200])
        .build();

    let c = graph
        .input("c")
        .with_dtype(DType::F32)
        .with_shape(vec![100, 200])
        .build();

    // (a + b) * c - 単純な演算チェーン
    let add = a + b;
    let mul = add * c;
    graph.output("result", mul);

    println!("【1/3】最適化なしでコンパイル...");
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline_no_opt = GenericPipeline::new(renderer, compiler);

    match pipeline_no_opt.compile_graph(graph.clone()) {
        Ok(_kernel) => println!("  ✓ コンパイル成功（最適化なし）\n"),
        Err(e) => println!("  ✗ エラー: {}\n", e),
    }

    println!("【2/3】グラフ最適化のみ有効でコンパイル...");
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline_graph_opt =
        GenericPipeline::new(renderer, compiler).with_graph_optimization(true);

    match pipeline_graph_opt.compile_graph(graph.clone()) {
        Ok(_kernel) => println!("  ✓ コンパイル成功（グラフ最適化あり）\n"),
        Err(e) => println!("  ✗ エラー: {}\n", e),
    }

    println!("【3/3】全最適化を有効にしてコンパイル...");
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();

    // カスタム設定で最適化を有効化
    let graph_config = GraphOptimizationConfig {
        beam_width: 5,
        max_steps: 10,
        show_progress: true,
    };

    let ast_config = AstOptimizationConfig {
        rule_max_iterations: 50,
        beam_width: 5,
        max_steps: 10,
        show_progress: true,
    };

    let mut pipeline_all_opt = GenericPipeline::new(renderer, compiler)
        .with_graph_optimization_config(graph_config)
        .with_ast_optimization_config(ast_config);

    match pipeline_all_opt.compile_graph(graph) {
        Ok(_kernel) => println!("\n  ✓ コンパイル成功（全最適化あり）\n"),
        Err(e) => println!("\n  ✗ エラー: {}\n", e),
    }

    println!("=== デモ完了 ===");
    println!("\nGenericPipelineに以下の最適化が組み込まれました：");
    println!("  1. グラフ最適化:");
    println!("     - ViewInsertion（Transpose含む）");
    println!("     - Fusion（演算融合）");
    println!("     - ParallelStrategy（並列化戦略）");
    println!("     - SimdSuggester（SIMD化）");
    println!("  2. AST最適化:");
    println!("     - ルールベース最適化（代数的簡約）");
    println!("     - ビームサーチ最適化");
}
