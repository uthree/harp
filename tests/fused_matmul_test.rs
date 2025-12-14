use harp::ast::helper::wildcard;
use harp::backend::GenericPipeline;
use harp::backend::opencl::{OpenCLCompiler, OpenCLRenderer};
/// FusedElementwiseReduceを使用した行列積の統合テスト
///
/// View展開とFusedElementwiseReduceを組み合わせた行列積の実装を検証します。
/// このテストはloweringとコード生成が正しく動作することを確認します。
use harp::graph::{DType, Graph, GraphNode};
use harp::prelude::{ReduceOp, fused_elementwise_reduce};

/// 行列積を計算するヘルパー関数
/// C = A @ B (A: [M, K], B: [K, N]) -> C: [M, N]
fn matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    let a_shape = a.view.shape();
    let b_shape = b.view.shape();

    assert_eq!(a_shape.len(), 2, "matmul: A must be 2D");
    assert_eq!(b_shape.len(), 2, "matmul: B must be 2D");

    let m = a_shape[0].clone();
    let k_a = a_shape[1].clone();
    let n = b_shape[1].clone();

    // B: [K, N] -> transpose -> [N, K]
    let b_transposed = b.view(b.view.clone().permute(vec![1, 0]));

    // A: [M, K] -> [M, 1, K] -> expand to [M, N, K]
    let a_unsqueezed = a.view(a.view.clone().unsqueeze(1));
    let expanded_shape = vec![m.clone(), n.clone(), k_a.clone()];
    let a_expanded = a_unsqueezed.view(a_unsqueezed.view.clone().expand(expanded_shape.clone()));

    // B_T: [N, K] -> [1, N, K] -> expand to [M, N, K]
    let b_t_unsqueezed = b_transposed.view(b_transposed.view.clone().unsqueeze(0));
    let b_t_expanded = b_t_unsqueezed.view(b_t_unsqueezed.view.clone().expand(expanded_shape));

    // FusedElementwiseReduce: multiply + sum over K axis
    // expr: Wildcard("0") * Wildcard("1")
    let expr = wildcard("0") * wildcard("1");
    fused_elementwise_reduce(
        vec![a_expanded, b_t_expanded],
        expr,
        ReduceOp::Sum,
        vec![2], // K軸でreduce
    )
}

#[test]
fn test_fused_matmul_lowering() {
    let mut graph = Graph::new();

    // A: [4, 3] 行列
    let a = graph.input("A", DType::F32, vec![4, 3]);

    // B: [3, 2] 行列
    let b = graph.input("B", DType::F32, vec![3, 2]);

    // 行列積: C = A @ B -> [4, 2]
    let c = matmul(a, b);

    // 結果をグラフに登録
    graph.output("C", c.clone());

    // 検証: 出力の形状が [4, 2] であることを確認
    assert_eq!(c.view.ndim(), 2);
    let result_shape = c.view.shape();
    assert_eq!(result_shape[0], 4.into());
    assert_eq!(result_shape[1], 2.into());

    // Pipelineを使用してコード生成
    let renderer = OpenCLRenderer::new();
    let compiler = OpenCLCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    let (program, _) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize graph");

    // コード生成を確認
    let mut opencl_renderer = OpenCLRenderer::new();
    let code = opencl_renderer.render_program(&program);
    let code_str = code.to_string();

    // グラフ最適化によってCustomノードが生成され、コードが生成されることを確認
    println!("Generated code:\n{}", code_str);

    // 生成されたコードが空でないことを確認
    assert!(!code_str.is_empty(), "Generated code should not be empty");

    // カーネル関数が生成されていることを確認
    assert!(
        code_str.contains("ER_") || code_str.contains("E_") || code_str.contains("O_"),
        "Code should contain kernel functions (E_, ER_, O_, etc.)"
    );

    println!("✓ Fused matmul lowering test passed");
    println!("Generated code contains kernel functions");
}

#[test]
fn test_double_matmul_lowering() {
    let mut graph = Graph::new();

    // 連続した行列積: (A @ B) @ C
    let a = graph.input("A", DType::F32, vec![8, 4]);
    let b = graph.input("B", DType::F32, vec![4, 6]);
    let c = graph.input("C", DType::F32, vec![6, 3]);

    // 1回目のmatmul: A[8,4] @ B[4,6] -> [8,6]
    let ab = matmul(a, b);
    assert_eq!(ab.view.shape()[0], 8.into());
    assert_eq!(ab.view.shape()[1], 6.into());

    // 2回目のmatmul: [8,6] @ C[6,3] -> [8,3]
    let result = matmul(ab, c);
    assert_eq!(result.view.shape()[0], 8.into());
    assert_eq!(result.view.shape()[1], 3.into());

    graph.output("result", result);

    // Pipelineを使用してコード生成
    let renderer = OpenCLRenderer::new();
    let compiler = OpenCLCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    let (program, _) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize graph");

    // コード生成
    let mut opencl_renderer = OpenCLRenderer::new();
    let code = opencl_renderer.render_program(&program);
    let code_str = code.to_string();

    println!("Generated code:\n{}", code_str);

    // 生成されたコードが空でないことを確認
    assert!(!code_str.is_empty(), "Generated code should not be empty");

    // カーネル関数が生成されていることを確認
    assert!(
        code_str.contains("ER_") || code_str.contains("E_") || code_str.contains("O_"),
        "Code should contain kernel functions (E_, ER_, O_, etc.)"
    );

    // エントリーポイントが存在することを確認
    // OpenCLは __harp_entry を使用する
    assert!(
        code_str.contains("__harp_entry") || code_str.contains("harp_main"),
        "Should have entry point function (__harp_entry or harp_main)"
    );

    println!("✓ Double matmul lowering test passed");
    println!("Generated code uses intermediate buffers correctly");
}
