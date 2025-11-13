//! GenericPipelineを使った最適化と可視化のデモ
//!
//! 行列積を含む複雑な計算グラフを作成し、GenericPipelineで最適化を実行して
//! その過程を可視化します。

use harp::backend::openmp::{CCompiler, CRenderer};
use harp::backend::{AstOptimizationConfig, GenericPipeline, GraphOptimizationConfig};
use harp::graph::{DType, Graph, GraphNode};
use harp_viz::HarpVizApp;

fn main() -> eframe::Result {
    // env_logger::init()の代わりにlog_captureを使う
    harp::opt::log_capture::init_with_env_logger();

    println!("=== Harp GenericPipeline 総合最適化デモ ===\n");
    println!("このデモでは、様々な最適化が適用される複雑な計算グラフを構築します。");
    println!("以下の最適化が順次適用されます：");
    println!("  1. グラフ最適化:");
    println!("     - View挿入 (転置の最適化)");
    println!("     - 演算の融合 (Fusion)");
    println!("  2. AST最適化:");
    println!("     - 定数畳み込み (Constant Folding)");
    println!("     - 代数的簡約 (x+0→x, x*1→x)");
    println!("     - ループ最適化 (タイル化、展開)");
    println!("最適化の各ステップがGenericPipelineに記録され、可視化されます。\n");

    // GenericPipelineを作成（最適化を組み込み）
    println!("【1/3】GenericPipelineを初期化中（最適化を組み込み）...");
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();

    // グラフ最適化の設定
    let graph_config = GraphOptimizationConfig {
        beam_width: 4,
        max_steps: 100,
        show_progress: true,
    };

    // AST最適化の設定
    let ast_config = AstOptimizationConfig {
        beam_width: 4,
        max_steps: 10000,
        show_progress: true,
    };

    let mut pipeline = GenericPipeline::new(renderer, compiler)
        .with_graph_optimization_config(graph_config)
        .with_ast_optimization_config(ast_config);

    println!("  ✓ Pipeline作成完了（最適化有効）\n");

    // 複雑な計算グラフを作成（複数の演算を含む）
    println!("【2/3】複雑な計算グラフを構築中...");
    let graph = create_complex_computation_graph();
    println!("  ✓ グラフ作成完了");
    println!("    - 入力数: {}", graph.inputs().len());
    println!("    - 出力数: {}", graph.outputs().len());
    println!();

    // 最適化を一括実行（コンパイルなし）
    println!("【3/3】最適化を実行中...");
    println!("  - グラフ最適化中...");
    let (_optimized_program, function_histories) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize graph");

    println!("  ✓ 最適化完了");
    println!(
        "    - グラフ最適化ステップ数: {}",
        pipeline
            .last_graph_optimization_history()
            .map_or(0, |h| h.len())
    );
    println!(
        "    - AST最適化ステップ数: {}",
        function_histories.values().map(|h| h.len()).sum::<usize>()
    );

    // デバッグ: ログがキャプチャされているか確認
    if let Some(graph_history) = pipeline.last_graph_optimization_history() {
        println!("  - グラフ最適化履歴のログ確認:");
        for (i, snapshot) in graph_history.snapshots().iter().take(3).enumerate() {
            println!("    Step {}: {} logs captured", i, snapshot.logs.len());
            if !snapshot.logs.is_empty() {
                println!("      First log: {}", snapshot.logs[0]);
            }
        }
    }

    if let Some((_, history)) = function_histories.iter().next() {
        println!("  - AST最適化履歴のログ確認:");
        for (i, snapshot) in history.snapshots().iter().take(3).enumerate() {
            println!("    Step {}: {} logs captured", i, snapshot.logs.len());
            if !snapshot.logs.is_empty() {
                println!("      First log: {}", snapshot.logs[0]);
            }
        }
    }

    println!();

    // 最適化の統計情報を表示
    print_optimization_stats(&pipeline);

    // 可視化アプリケーションを起動
    println!("\n可視化UIを起動中...");
    println!("  - Graph Viewerタブ: グラフ最適化の履歴を表示");
    println!("  - AST Viewerタブ: AST最適化の履歴を表示");
    println!("  - 矢印キーで前後のステップに移動できます");
    println!();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("Harp Pipeline Optimization Visualizer - Comprehensive Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Pipeline Visualizer",
        options,
        Box::new(move |_cc| {
            let mut app = HarpVizApp::new();

            // グラフ最適化履歴を読み込む
            if let Some(graph_history) = pipeline.last_graph_optimization_history() {
                app.load_graph_optimization_history(graph_history.clone());
            }

            // 複数のFunction最適化履歴を読み込む
            app.load_multiple_ast_histories(function_histories);

            Ok(Box::new(app))
        }),
    )
}

/// 行列積を計算するヘルパー関数
/// C = A @ B (A: [M, K], B: [K, N]) -> C: [M, N]
fn matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    use harp::graph::shape::Expr;

    let a_shape = a.view.shape();
    let b_shape = b.view.shape();

    assert_eq!(a_shape.len(), 2, "matmul: A must be 2D");
    assert_eq!(b_shape.len(), 2, "matmul: B must be 2D");

    let m = a_shape[0].clone();
    let k_a = a_shape[1].clone();
    let k_b = b_shape[0].clone();
    let n = b_shape[1].clone();

    // K次元が一致するか確認（数値の場合のみ）
    if let (Expr::Const(k_a_val), Expr::Const(k_b_val)) = (&k_a, &k_b) {
        assert_eq!(
            k_a_val, k_b_val,
            "matmul: dimension mismatch K: {} != {}",
            k_a_val, k_b_val
        );
    }

    // A: [M, K] -> [M, K, 1]
    let a_unsqueezed = a.view(a.view.clone().unsqueeze(2));

    // B: [K, N] -> [1, K, N]
    let b_unsqueezed = b.view(b.view.clone().unsqueeze(0));

    // expand to [M, K, N]
    let expanded_shape = vec![m.clone(), k_a.clone(), n.clone()];
    let a_expanded = a_unsqueezed.view(a_unsqueezed.view.clone().expand(expanded_shape.clone()));
    let b_expanded = b_unsqueezed.view(b_unsqueezed.view.clone().expand(expanded_shape));

    // elementwise multiply: [M, K, N]
    let product = a_expanded * b_expanded;

    // reduce_sum along axis=1: [M, N]
    product.reduce_sum(1)
}

/// 複雑な計算グラフを作成（行列積と定数演算を含む）
///
/// 以下の計算を実装:
/// ```
/// # 行列積と定数演算を組み合わせた計算
/// # 複数の最適化が順次適用される
///
/// # 定数畳み込み
/// scale = 2.0 * 3.0  # 6.0に最適化
///
/// # 行列積 (View挿入とFusionが働く)
/// temp1 = matmul(a, b)  # [M, K] @ [K, N] -> [M, N]
///
/// # Elementwise演算の連鎖 (Fusionが働く)
/// temp2 = temp1 + c
/// temp3 = temp2 * scale
///
/// # さらなる行列積
/// result = matmul(temp3, d)  # [M, N] @ [N, P] -> [M, P]
/// ```
fn create_complex_computation_graph() -> Graph {
    let mut graph = Graph::new();

    // サイズは小さめにして最適化の効果を見やすくする
    let m = 64;
    let k = 32;
    let n = 64;
    let p = 128;

    // 入力行列
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![m, k])
        .build();

    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![k, n])
        .build();

    let c = graph
        .input("c")
        .with_dtype(DType::F32)
        .with_shape(vec![m, n])
        .build();

    let d = graph
        .input("d")
        .with_dtype(DType::F32)
        .with_shape(vec![n, p])
        .build();

    // 定数演算（定数畳み込みが働く）
    let const1 = GraphNode::constant(2.0);
    let const2 = GraphNode::constant(3.0);
    let scale_scalar = const1 * const2; // 6.0に畳み込まれる

    // スカラーを [M, N] にブロードキャスト
    let scale_unsqueezed = scale_scalar.view(scale_scalar.view.clone().unsqueeze(0).unsqueeze(0));
    let scale = scale_unsqueezed.view(
        scale_unsqueezed
            .view
            .clone()
            .expand(vec![m.into(), n.into()]),
    );

    // 行列積 (View挿入とFusionが働く)
    let temp1 = matmul(a, b); // [M, N]

    // Elementwise演算の連鎖 (これらがFusionで1つのカーネルになる)
    let temp2 = temp1 + c;
    let temp3 = temp2 * scale;

    // さらなる行列積
    let result = matmul(temp3, d); // [M, P]

    // 出力
    graph.output("result", result);

    graph
}

/// 最適化の統計情報を表示
fn print_optimization_stats(pipeline: &GenericPipeline<CRenderer, CCompiler>) {
    println!("=== 最適化統計 ===");

    if let Some(graph_history) = pipeline.last_graph_optimization_history() {
        println!("\n【グラフ最適化】");
        println!("  ステップ数: {}", graph_history.len());

        if let (Some(first), Some(last)) = (
            graph_history.get(0),
            graph_history.get(graph_history.len() - 1),
        ) {
            let cost_reduction = first.cost - last.cost;
            let cost_reduction_percent = if first.cost > 0.0 {
                (cost_reduction / first.cost) * 100.0
            } else {
                0.0
            };

            println!("  初期コスト: {:.2}", first.cost);
            println!("  最終コスト: {:.2}", last.cost);
            println!(
                "  コスト削減: {:.2} ({:.1}%)",
                cost_reduction, cost_reduction_percent
            );
        }

        // コスト遷移を表示
        println!("\n  コスト遷移:");
        for i in 0..graph_history.len().min(5) {
            if let Some(snapshot) = graph_history.get(i) {
                println!("    Step {}: {:.2}", snapshot.step, snapshot.cost);
            }
        }
        if graph_history.len() > 5 {
            println!("    ... ({} steps total)", graph_history.len());
        }
    }

    if let Some(ast_history) = pipeline.last_ast_optimization_history() {
        println!("\n【AST最適化】");
        println!("  ステップ数: {}", ast_history.len());

        if let (Some(first), Some(last)) = (
            ast_history.snapshots().first(),
            ast_history.snapshots().last(),
        ) {
            let cost_reduction = first.cost - last.cost;
            let cost_reduction_percent = if first.cost > 0.0 {
                (cost_reduction / first.cost) * 100.0
            } else {
                0.0
            };

            println!("  初期コスト: {:.2}", first.cost);
            println!("  最終コスト: {:.2}", last.cost);
            println!(
                "  コスト削減: {:.2} ({:.1}%)",
                cost_reduction, cost_reduction_percent
            );
        }
    }
}
