//! GenericPipelineを使った最適化と可視化のデモ
//!
//! 行列積を含む複雑な計算グラフを作成し、GenericPipelineで最適化を実行して
//! その過程を可視化します。

use harp::backend::openmp::{CCompiler, CRenderer};
use harp::backend::{GenericPipeline, OptimizationConfig};
use harp::graph::{DType, Graph, GraphNode};
use harp::prelude::{FusedElementwiseOp, FusedInput};
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

    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 最適化を有効化（フィールドに直接アクセス）
    pipeline.enable_graph_optimization = true;
    pipeline.enable_ast_optimization = true;

    // グラフ最適化の設定
    pipeline.graph_config = OptimizationConfig {
        beam_width: 4,
        max_steps: 100,
        show_progress: true,
    };

    // AST最適化の設定
    pipeline.ast_config = OptimizationConfig {
        beam_width: 4,
        max_steps: 10000,
        show_progress: true,
    };

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
    let (optimized_program, function_histories) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize graph");

    println!("  ✓ 最適化完了");

    // 生成されたCコードを表示
    println!("\n=== 生成されたCコード ===");
    let mut c_renderer = CRenderer::new();
    let code = c_renderer.render_program(&optimized_program);
    println!("{}", code);
    println!(
        "    - グラフ最適化ステップ数: {}",
        pipeline.histories.graph.as_ref().map_or(0, |h| h.len())
    );
    println!(
        "    - AST最適化ステップ数: {}",
        function_histories.values().map(|h| h.len()).sum::<usize>()
    );

    // デバッグ: ログがキャプチャされているか確認
    if let Some(graph_history) = &pipeline.histories.graph {
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
            if let Some(graph_history) = &pipeline.histories.graph {
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
    use harp::graph::ops::fused_elementwise_reduce;
    use harp::graph::shape::Expr;
    use harp::graph::{ElementwiseOp, ReduceOp};

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

    // A: [M, K]をそのまま使用
    // B: [K, N] -> transpose -> [N, K] として扱う

    // 結果は [M, N] で、各 (i, j) について sum over k of A[i, k] * B[k, j]
    // FusedElementwiseReduceを使用: fused_elementwise_reduce(inputs, op, reduce_op, axis)

    // ただし、現状のfused_elementwise_reduceは入力が同じshapeである必要があるため、
    // ここでは2次元のreduceパターンを直接実装

    // [M, K] と [K, N] を組み合わせる方法:
    // 出力 [M, N] の各要素について、K次元でsum(A[m, k] * B[k, n])を計算

    // 簡易実装: A の各行と B の各列の内積
    // 現状のLowererの制限により、単純なFusedElementwiseReduceを使用

    // A: [M, K] -> [M, K] (そのまま)
    // B: [K, N] -> [K, N] (転置して [N, K] として扱うが、Viewで対応)

    // fused_elementwise_reduceを使うには、すべての入力が同じshapeである必要がある
    // そのため、3次元に拡張するアプローチは現状のLowererでは未サポート

    // 代わりに、GraphOp::Reduceを使用してシンプルに実装
    // A: [M, K], B: [K, N]
    // 1. Bを転置: [N, K]
    // 2. Aを[M, 1, K]にexpand, Bを[1, N, K]にexpand -> broadcast後 [M, N, K]
    // 3. elementwise multiply
    // 4. reduce_sum(axis=2) -> [M, N]

    // この処理はブロードキャストを必要とするため、現在のLowererでは直接サポートされていない
    // 暫定的に、fused_elementwise_reduceを使った近似を行う

    // 仮実装: 各出力位置について手動で計算するパターン（Lowerer拡張が必要）
    // 現状では、単純なreduce_sumのパターンを使用

    // 暫定的な実装: 2次元行列積を直接サポートするGraphOpが必要
    // 今は、reduce操作を使った近似

    // 簡単な回避策: Aの転置を使って内積を計算
    // A: [M, K], B: [K, N]
    // 転置B: [N, K]

    // まず単純にB[K, N]をviewで[N, K]に転置
    let b_transposed = b.view(b.view.clone().permute(vec![1, 0])); // [N, K]

    // 出力は[M, N]になる
    // 各(m, n)について: sum_k A[m, k] * B_T[n, k]

    // これをfused_elementwise_reduceで実現するには、
    // A: [M, K] を [M, 1, K] に拡張
    // B_T: [N, K] を [1, N, K] に拡張
    // broadcast後 [M, N, K]
    // multiply -> [M, N, K]
    // reduce_sum(axis=2) -> [M, N]

    let a_unsqueezed = a.view(a.view.clone().unsqueeze(1)); // [M, 1, K]
    let b_t_unsqueezed = b_transposed.view(b_transposed.view.clone().unsqueeze(0)); // [1, N, K]

    let expanded_shape = vec![m.clone(), n.clone(), k_a.clone()];
    let a_expanded = a_unsqueezed.view(a_unsqueezed.view.clone().expand(expanded_shape.clone()));
    let b_t_expanded = b_t_unsqueezed.view(b_t_unsqueezed.view.clone().expand(expanded_shape));

    // FusedElementwiseReduceを使用
    fused_elementwise_reduce(
        vec![a_expanded, b_t_expanded],
        vec![FusedElementwiseOp {
            op: ElementwiseOp::Mul,
            inputs: vec![FusedInput::GraphInput(0), FusedInput::GraphInput(1)],
        }],
        ReduceOp::Sum,
        2, // K軸でreduce
    )
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
    let n = 16;
    let p = 8;

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

    // 1回目の行列積: matmul(a, b) -> [M, N]
    let temp1 = matmul(a, b);

    // Elementwise演算の連鎖
    let temp2 = temp1 + c; // [M, N]

    // 2回目の行列積: matmul(temp2, d) -> [M, P]
    let result = matmul(temp2, d);
    graph.output("result", result);

    graph
}

/// 最適化の統計情報を表示
fn print_optimization_stats(pipeline: &GenericPipeline<CRenderer, CCompiler>) {
    println!("=== 最適化統計 ===");

    if let Some(graph_history) = &pipeline.histories.graph {
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

    if let Some(ast_history) = &pipeline.histories.ast {
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
