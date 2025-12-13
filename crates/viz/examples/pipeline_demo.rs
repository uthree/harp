//! GenericPipelineを使った最適化と可視化のデモ（OpenCL版）
//!
//! 行列積を含む複雑な計算グラフを作成し、GenericPipelineで最適化を実行して
//! その過程を可視化します。OpenCLバックエンドを使用してGPU向けコードを生成します。
//!
//! ## 実測値を使ったAST最適化
//! このデモでは、静的コスト推定に加えて、RuntimeSelectorを使った実測値ベースの
//! AST最適化も行います。静的コストで足切りした後、実際にカーネルを実行して
//! 実行時間を計測し、最適なAST変換を選択します。

use harp::ast::helper::wildcard;
use harp::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLRenderer};
use harp::backend::pipeline::create_ast_loop_suggester;
use harp::backend::{GenericPipeline, KernelSignature, OptimizationConfig};
use harp::graph::{DType, Graph, GraphNode};
use harp::lowerer::create_signature;
use harp::opt::ast::rules::all_algebraic_rules;
use harp::opt::ast::{
    BeamSearchOptimizer as AstBeamSearchOptimizer, CostEstimator, Optimizer, RuleBaseOptimizer,
    SimpleCostEstimator,
};
use harp::opt::selector::RuntimeSelector;
use harp_viz::{HarpVizApp, RendererType};

fn main() -> eframe::Result {
    // env_logger::init()の代わりにlog_captureを使う
    harp::opt::log_capture::init_with_env_logger();

    println!("=== Harp GenericPipeline 統合最適化デモ (OpenCL版) ===\n");
    println!("このデモでは、様々な最適化が適用される複雑な計算グラフを構築します。");
    println!("OpenCLバックエンドを使用してGPU向けコードを生成します。");
    println!("統合最適化により以下が一括で適用されます：");
    println!("  - View挿入 (転置の最適化)");
    println!("  - 演算の融合 (Fusion)");
    println!("  - 並列化戦略の最適化 (GPU用)");
    println!("  - Lowering (GraphOp → Kernel変換)");
    println!("  - カーネルマージ (複数Kernel → Program統合)");
    println!("  - AST最適化 (代数的簡約、ループ最適化)");
    println!("  - **実測値ベースのAST最適化 (RuntimeSelector)**");
    println!("最適化の各ステップがGenericPipelineに記録され、可視化されます。\n");

    // GenericPipelineを作成（最適化を組み込み）
    println!("【1/3】GenericPipelineを初期化中（OpenCLバックエンド）...");
    let renderer = OpenCLRenderer::new();
    let compiler = OpenCLCompiler::new();

    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // マルチフェーズ最適化を有効化（グラフ準備 → Lowering）
    pipeline.collect_histories = true;

    // グラフ最適化の設定
    pipeline.graph_config = OptimizationConfig {
        beam_width: 4,
        max_steps: 10000,
        show_progress: true,
        early_termination_threshold: None, // 早期終了を無効化
    };

    // AST最適化の設定
    pipeline.ast_config = OptimizationConfig {
        beam_width: 4,
        max_steps: 10000,
        show_progress: true,
        early_termination_threshold: None,
    };

    println!("  ✓ Pipeline作成完了（統合最適化有効）\n");

    // 複雑な計算グラフを作成（複数の演算を含む）
    println!("【2/4】複雑な計算グラフを構築中...");
    let graph = create_complex_computation_graph();
    println!("  ✓ グラフ作成完了");
    println!("    - 入力数: {}", graph.input_metas().len());
    println!("    - 出力数: {}", graph.outputs().len());

    // シグネチャを作成（RuntimeSelectorで使用）
    let signature = create_signature(&graph);
    println!(
        "    - シグネチャ作成完了（入力: {}, 出力: {}）",
        signature.inputs.len(),
        signature.outputs.len()
    );
    println!();

    // グラフ最適化を実行（AST最適化は無効化）
    println!("【3/4】グラフ最適化を実行中...");
    println!("  - 統合グラフ最適化中（AST最適化は後で実測値ベースで実行）...");
    pipeline.enable_ast_optimization = false; // AST最適化を無効化
    let (static_optimized_program, _) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize graph");
    println!("  ✓ グラフ最適化完了");

    // RuntimeSelectorを使ったAST最適化
    println!("\n【4/4】実測値ベースのAST最適化を実行中...");
    let (optimized_program, ast_histories) = optimize_ast_with_runtime_selector(
        static_optimized_program.clone(),
        signature,
        &pipeline.ast_config,
    );
    println!("  ✓ 実測値ベースAST最適化完了");

    // 生成されたOpenCLコードを表示
    println!("\n=== 生成されたOpenCLコード ===");
    let mut opencl_renderer = OpenCLRenderer::new();
    let code = opencl_renderer.render_program(&optimized_program);
    println!("{}", code);

    // 統合最適化の履歴情報を表示
    let graph_steps = pipeline.histories.graph.as_ref().map_or(0, |h| h.len());
    println!("    - グラフ最適化ステップ数: {}", graph_steps);

    // デバッグ: 最終グラフのProgramRoot.srcを確認
    if let Some(graph_history) = &pipeline.histories.graph {
        if let Some(final_snapshot) = graph_history.snapshots().last() {
            if let Some(sink) = final_snapshot.graph.program_root() {
                println!("\n=== Final Graph ProgramRoot.src (recursive) ===");
                fn print_node(
                    node: &harp::graph::GraphNode,
                    depth: usize,
                    visited: &mut std::collections::HashSet<*const harp::graph::GraphNodeData>,
                ) {
                    let ptr = node.as_ptr();
                    if visited.contains(&ptr) {
                        println!("{}(already visited)", "  ".repeat(depth));
                        return;
                    }
                    visited.insert(ptr);

                    let op_name = match &node.op {
                        harp::graph::GraphOp::Buffer { name } => format!("Buffer({})", name),
                        harp::graph::GraphOp::Kernel { input_buffers, .. } => {
                            let buffers = input_buffers
                                .as_ref()
                                .map(|b| b.iter().map(|m| m.name.clone()).collect::<Vec<_>>());
                            format!("Kernel(input_buffers={:?})", buffers)
                        }
                        harp::graph::GraphOp::ProgramRoot { outputs, .. } => {
                            format!("ProgramRoot(outputs={:?})", outputs)
                        }
                        harp::graph::GraphOp::View(_) => "View".to_string(),
                        _ => format!("{:?}", std::mem::discriminant(&node.op)),
                    };
                    println!(
                        "{}{} (src_count={})",
                        "  ".repeat(depth),
                        op_name,
                        node.src.len()
                    );
                    for src in &node.src {
                        print_node(src, depth + 1, visited);
                    }
                }

                println!("ProgramRoot src count: {}", sink.src.len());
                let mut visited = std::collections::HashSet::new();
                for (i, src) in sink.src.iter().enumerate() {
                    println!("src[{}]:", i);
                    print_node(src, 1, &mut visited);
                }
            }
        }
    }

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

    println!();

    // 最適化の統計情報を表示
    print_optimization_stats(&pipeline);

    // 可視化アプリケーションを起動
    println!("\n可視化UIを起動中...");
    println!("  - Graph Viewerタブ: グラフ最適化の履歴を表示");
    println!("  - Code Viewerタブ: AST最適化の履歴と最終コードを表示");
    println!("  - 矢印キーで前後のステップに移動できます");
    println!();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("Harp Pipeline Optimization Visualizer - Unified Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Pipeline Visualizer",
        options,
        Box::new(move |_cc| {
            // OpenCLレンダラータイプでHarpVizAppを作成
            let mut app = HarpVizApp::with_renderer_type(RendererType::OpenCL);

            // グラフ最適化履歴を読み込む（Phase 1 + Phase 2を結合）
            if let Some(graph_history) = pipeline.histories.combined_graph_history() {
                app.load_graph_optimization_history(graph_history);
            }

            // AST最適化履歴を読み込む（Code ViewerでAST最適化の各ステップを表示可能）
            // RuntimeSelectorを使った実測値ベースの最適化履歴
            app.load_ast_optimization_history(ast_histories);

            // AST最適化済みのProgramをCode Viewerに直接ロード
            // これにより、最終コード表示モードではAST最適化後のコードが表示される
            app.load_optimized_ast(optimized_program);

            Ok(Box::new(app))
        }),
    )
}

/// 行列積を計算するヘルパー関数
/// C = A @ B (A: [M, K], B: [K, N]) -> C: [M, N]
fn matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    use harp::graph::ops::fused_elementwise_reduce;
    use harp::graph::shape::Expr;
    use harp::graph::ReduceOp;

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
    // expr: Wildcard("0") * Wildcard("1")
    let expr = wildcard("0") * wildcard("1");
    fused_elementwise_reduce(
        vec![a_expanded, b_t_expanded],
        expr,
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
    let a = graph.input("a", DType::F32, vec![m, k]);

    let b = graph.input("b", DType::F32, vec![k, n]);

    let c = graph.input("c", DType::F32, vec![m, n]);

    let d = graph.input("d", DType::F32, vec![n, p]);

    // 1回目の行列積: matmul(a, b) -> [M, N]
    let temp1 = matmul(a, b);

    // Elementwise演算の連鎖
    let temp2 = temp1 + c; // [M, N]

    // 2回目の行列積: matmul(temp2, d) -> [M, P]
    let x = matmul(temp2, d);

    // 複数の出力ノードと定数の計算
    let y = &x + 3.0f32;
    let y = &y + 5.0f32;
    let y = &y + 8.0f32;
    graph.output("x", x);
    graph.output("y", y);

    graph
}

/// 2段階AST最適化を実行
///
/// 第1段階: ルールベース最適化（高速、ビームサーチなし）
///   - 定数畳み込み、簡約化、正規化
///   - RuleBaseOptimizerを直接使用（パターンマッチングの収束まで適用）
///
/// 第2段階: 実測値ベース最適化（RuntimeSelector）
///   - ループ最適化などの構造変換
///   - 実際にカーネルを実行して最適な変換を選択
fn optimize_ast_with_runtime_selector(
    program: harp::ast::AstNode,
    signature: KernelSignature,
    config: &OptimizationConfig,
) -> (harp::ast::AstNode, harp::opt::ast::OptimizationHistory) {
    use harp::ast::DType as AstDType;
    use harp::graph::shape::Expr;
    use harp::opt::ast::history::{OptimizationHistory, OptimizationSnapshot};

    // ==========================================================
    // 第1段階: ルールベース最適化（高速、ビームサーチなし）
    // ==========================================================
    println!("\n  【第1段階】ルールベース最適化（パターンマッチング）");

    // 代数的簡約ルールを取得
    let rules = all_algebraic_rules();
    println!("    - {} 個のルールを使用", rules.len());

    // RuleBaseOptimizerを直接使用（ビームサーチなし、高速）
    let rule_optimizer = RuleBaseOptimizer::new(rules).with_max_iterations(100);

    let initial_cost = SimpleCostEstimator::new().estimate(&program);
    let rule_optimized = rule_optimizer.optimize(program.clone());
    let after_rule_cost = SimpleCostEstimator::new().estimate(&rule_optimized);

    println!(
        "    - コスト: {:.2} → {:.2} ({:.1}% 削減)",
        initial_cost,
        after_rule_cost,
        if initial_cost > 0.0 {
            (initial_cost - after_rule_cost) / initial_cost * 100.0
        } else {
            0.0
        }
    );

    // 履歴を手動で作成（RuleBaseOptimizerは履歴を返さないため）
    let mut combined_history = OptimizationHistory::new();
    combined_history.add_snapshot(OptimizationSnapshot::new(
        0,
        program,
        initial_cost,
        "Initial AST".to_string(),
        0,
        None,
    ));
    combined_history.add_snapshot(OptimizationSnapshot::new(
        1,
        rule_optimized.clone(),
        after_rule_cost,
        "After rule-based optimization".to_string(),
        0,
        Some("RuleBaseOptimizer".to_string()),
    ));

    // ==========================================================
    // 第2段階: 実測値ベース最適化（RuntimeSelector）
    // ==========================================================
    println!("\n  【第2段階】実測値ベース最適化（RuntimeSelector）");

    // バッファファクトリを作成
    let buffer_factory = move |sig: &KernelSignature| -> Vec<OpenCLBuffer> {
        sig.inputs
            .iter()
            .chain(sig.outputs.iter())
            .map(|buf_sig| {
                let shape: Vec<usize> = buf_sig
                    .shape
                    .iter()
                    .map(|expr| match expr {
                        Expr::Const(c) => (*c).max(1) as usize,
                        _ => 1,
                    })
                    .collect();
                OpenCLBuffer::with_dtype(shape, AstDType::F32)
            })
            .collect()
    };

    // RuntimeSelectorを作成（パラメータは控えめに）
    let runtime_selector = RuntimeSelector::new(
        OpenCLRenderer::new(),
        OpenCLCompiler::new(),
        signature,
        buffer_factory,
    )
    .with_pre_filter_count(3)
    .with_measurement_count(2);

    println!("    - 足切り: 3件, 計測回数: 2回");

    // ループ最適化などの構造変換用Suggester（RuleBaseSuggesterを除く）
    let loop_suggester = create_ast_loop_suggester();

    // 実測値ベース最適化（ステップ数は控えめに）
    let max_steps = config.max_steps;
    let runtime_optimizer = AstBeamSearchOptimizer::new(loop_suggester)
        .with_selector(runtime_selector)
        .with_beam_width(config.beam_width)
        .with_max_steps(max_steps)
        .with_progress(config.show_progress);

    println!(
        "    - ビーム幅: {}, 最大ステップ: {}",
        config.beam_width.min(2),
        max_steps
    );

    let (final_optimized, runtime_history) =
        runtime_optimizer.optimize_with_history(rule_optimized);

    println!("    - ステップ数: {}", runtime_history.len());

    // 履歴を結合（ルールベース → 実測値ベース）
    for snapshot in runtime_history.snapshots() {
        combined_history.add_snapshot(snapshot.clone());
    }

    (final_optimized, combined_history)
}

/// 最適化の統計情報を表示
fn print_optimization_stats(pipeline: &GenericPipeline<OpenCLRenderer, OpenCLCompiler>) {
    println!("=== 最適化統計 ===");

    // 統合グラフ最適化
    if let Some(graph_history) = &pipeline.histories.graph {
        println!("\n【統合グラフ最適化】");
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
}
