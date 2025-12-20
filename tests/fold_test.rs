//! Fold演算のテスト
//!
//! Fold演算（col2im）が正しくコンパイルされることを確認

#![cfg(feature = "opencl")]

use harp::backend::opencl::OpenCLRenderer;
use harp::backend::{MultiPhaseConfig, Renderer, create_multi_phase_optimizer};
use harp::lowerer::extract_program;
use harp::opt::graph::GraphOptimizer;
use harp::prelude::*;

/// コンパイルテストのヘルパー関数
fn compile_graph_test(graph: Graph, test_name: &str) {
    // Phase 1: Graph optimization
    let config = MultiPhaseConfig::new()
        .with_beam_width(4)
        .with_max_steps(1000)
        .with_progress(false)
        .with_collect_logs(false);

    let optimizer = create_multi_phase_optimizer(config);
    let (optimized_graph, _) = optimizer.optimize_with_history(graph);

    // Phase 2: Lower to AST
    let program = extract_program(optimized_graph);

    // Phase 3: Render to code
    let renderer = OpenCLRenderer::new();
    let code = renderer.render(&program);
    let code_str: String = code.into();

    // コードが生成されていることを確認
    assert!(
        !code_str.is_empty(),
        "{}: Generated code should not be empty",
        test_name
    );

    assert!(
        code_str.contains("__kernel") || code_str.contains("void"),
        "{}: Code should contain kernel functions",
        test_name
    );
}

#[test]
fn test_fold1d_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold1dのテスト: unfold -> fold で元に戻ることを確認
    let mut graph = Graph::new();

    // 入力: [2, 10] (C_in=2, L=10)
    let input = graph.input("input", DType::F32, vec![2, 10]);

    // unfold: kernel_size=3, stride=1, dilation=1, groups=1
    // 出力: [2, 3, 8]
    let unfolded = input.unfold(3, 1, 1, 1);

    // fold: 元に戻す
    // output_size=[2, 10], kernel_size=3, stride=1, dilation=1, groups=1
    let folded = unfolded.fold(vec![2, 10], 3, 1, 1, 1);

    graph.output("result", folded);

    compile_graph_test(graph, "Fold1d");
}

#[test]
fn test_fold2d_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold2dのテスト
    let mut graph = Graph::new();

    // 入力: [3, 5, 5] (C_in=3, H=5, W=5)
    let input = graph.input("input", DType::F32, vec![3, 5, 5]);

    // unfold: kernel_size=(3,3), stride=(1,1), dilation=(1,1), groups=1
    let unfolded = input.unfold((3, 3), (1, 1), (1, 1), 1);

    // fold: 元に戻す
    let folded = unfolded.fold(vec![3, 5, 5], (3, 3), (1, 1), (1, 1), 1);

    graph.output("result", folded);

    compile_graph_test(graph, "Fold2d");
}

#[test]
fn test_fold3d_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold3dのテスト
    let mut graph = Graph::new();

    // 入力: [2, 4, 4, 4] (C_in=2, D=4, H=4, W=4)
    let input = graph.input("input", DType::F32, vec![2, 4, 4, 4]);

    // unfold: kernel_size=(2,2,2), stride=(1,1,1), dilation=(1,1,1), groups=1
    let unfolded = input.unfold((2, 2, 2), (1, 1, 1), (1, 1, 1), 1);

    // fold: 元に戻す
    let folded = unfolded.fold(vec![2, 4, 4, 4], (2, 2, 2), (1, 1, 1), (1, 1, 1), 1);

    graph.output("result", folded);

    compile_graph_test(graph, "Fold3d");
}

#[test]
fn test_fold1d_groups_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold1d with groups=2 のテスト
    let mut graph = Graph::new();

    // 入力: [4, 10] (C_in=4, L=10)
    let input = graph.input("input", DType::F32, vec![4, 10]);

    // unfold with groups=2: [4, 10] -> [2, 2, 3, 8]
    // (groups, C/groups, k, L') = (2, 2, 3, 8)
    let unfolded = input.unfold(3, 1, 1, 2);

    // fold: 元に戻す
    let folded = unfolded.fold(vec![4, 10], 3, 1, 1, 2);

    graph.output("result", folded);

    compile_graph_test(graph, "Fold1d_groups");
}

#[test]
fn test_fold2d_groups_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold2d with groups=2 のテスト
    let mut graph = Graph::new();

    // 入力: [4, 8, 8] (C_in=4, H=8, W=8)
    let input = graph.input("input", DType::F32, vec![4, 8, 8]);

    // unfold with groups=2: [4, 8, 8] -> [2, 2, 3, 3, 6, 6]
    let unfolded = input.unfold((3, 3), (1, 1), (1, 1), 2);

    // fold: 元に戻す
    let folded = unfolded.fold(vec![4, 8, 8], (3, 3), (1, 1), (1, 1), 2);

    graph.output("result", folded);

    compile_graph_test(graph, "Fold2d_groups");
}

#[test]
fn test_fold2d_stride_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold2d with stride=2 のテスト
    let mut graph = Graph::new();

    // 入力: [3, 8, 8] (C_in=3, H=8, W=8)
    let input = graph.input("input", DType::F32, vec![3, 8, 8]);

    // unfold with stride=2: [3, 8, 8] -> [3, 3, 3, 3, 3]
    // L' = (8 - 3) / 2 + 1 = 3
    let unfolded = input.unfold((3, 3), (2, 2), (1, 1), 1);

    // fold: 元に戻す
    let folded = unfolded.fold(vec![3, 8, 8], (3, 3), (2, 2), (1, 1), 1);

    graph.output("result", folded);

    compile_graph_test(graph, "Fold2d_stride");
}

#[test]
fn test_fold2d_stride_groups_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold2d with stride=2 and groups=2 のテスト
    let mut graph = Graph::new();

    // 入力: [4, 8, 8] (C_in=4, H=8, W=8)
    let input = graph.input("input", DType::F32, vec![4, 8, 8]);

    // unfold with stride=2, groups=2
    let unfolded = input.unfold((3, 3), (2, 2), (1, 1), 2);

    // fold: 元に戻す
    let folded = unfolded.fold(vec![4, 8, 8], (3, 3), (2, 2), (1, 1), 2);

    graph.output("result", folded);

    compile_graph_test(graph, "Fold2d_stride_groups");
}
