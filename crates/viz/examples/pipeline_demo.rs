//! GenericPipelineを使った最適化と可視化のデモ（OpenCL版）
//!
//! 行列積を含む複雑な計算グラフを作成し、GenericPipelineで最適化を実行して
//! その過程を可視化します。OpenCLバックエンドを使用してGPU向けコードを生成します。
//!
//! ## 実測値を使ったAST最適化
//! このデモでは、GenericPipelineのRuntimeSelector機能を使用して、
//! 静的コスト推定に加えて実測値ベースのAST最適化を行います。

use harp::ast::helper::wildcard;
use harp::ast::DType as AstDType;
use harp::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLRenderer};
use harp::backend::{GenericPipeline, OptimizationConfig};
use harp::graph::shape::Expr;
use harp::graph::{DType, Graph, GraphNode, ReduceOp};
use harp_viz::{HarpVizApp, RendererType};

fn main() -> eframe::Result {
    // env_logger::init()の代わりにlog_captureを使う
    harp::opt::log_capture::init_with_env_logger();

    println!("=== Harp GenericPipeline 統合最適化デモ (OpenCL版) ===\n");
    println!("このデモでは、GenericPipelineのRuntimeSelector機能を使用して");
    println!("実測値ベースのAST最適化を行います。\n");
    println!("統合最適化により以下が一括で適用されます：");
    println!("  - View挿入 (転置の最適化)");
    println!("  - 演算の融合 (Fusion)");
    println!("  - 並列化戦略の最適化 (GPU用)");
    println!("  - Lowering (GraphOp -> Kernel変換)");
    println!("  - カーネルマージ (複数Kernel -> Program統合)");
    println!("  - AST最適化 (代数的簡約、ループ最適化)");
    println!("  - **実測値ベースのAST最適化 (RuntimeSelector)**");
    println!();

    // GenericPipelineを作成（最適化を組み込み）
    println!("[1/3] GenericPipelineを初期化中（OpenCLバックエンド）...");
    let renderer = OpenCLRenderer::new();
    let compiler = OpenCLCompiler::new();

    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 履歴収集を有効化
    pipeline.collect_histories = true;

    // グラフ最適化の設定
    pipeline.graph_config = OptimizationConfig {
        beam_width: 4,
        max_steps: 100,
        show_progress: true,
        early_termination_threshold: Some(10),
        pre_filter_count: 10,
        measurement_count: 10,
    };

    // AST最適化の設定（RuntimeSelector用パラメータも含む）
    pipeline.ast_config = OptimizationConfig {
        beam_width: 4,
        max_steps: 500,
        show_progress: true,
        early_termination_threshold: Some(10),
        pre_filter_count: 3,  // RuntimeSelector: 静的コストで3件に足切り
        measurement_count: 2, // RuntimeSelector: 2回計測して平均
    };

    // RuntimeSelector用のバッファファクトリを設定
    // これにより、AST最適化時にRuntimeSelectorが自動的に使用される
    pipeline.set_runtime_buffer_factory(|sig| {
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
    });

    println!("  - Pipeline作成完了");
    println!(
        "  - RuntimeSelector有効: pre_filter={}, measurement={}",
        pipeline.ast_config.pre_filter_count, pipeline.ast_config.measurement_count
    );
    println!();

    // 複雑な計算グラフを作成
    println!("[2/3] 複雑な計算グラフを構築中...");
    let graph = create_complex_computation_graph();
    println!("  - グラフ作成完了");
    println!("    - 入力数: {}", graph.input_metas().len());
    println!("    - 出力数: {}", graph.outputs().len());
    println!();

    // 最適化を実行（グラフ最適化 + RuntimeSelectorを使ったAST最適化）
    // set_runtime_buffer_factory()を呼び出しているため、自動的にRuntimeSelectorが使用される
    println!("[3/3] 最適化を実行中...");
    println!("  - グラフ最適化 + RuntimeSelector AST最適化を同時実行");
    let (optimized_program, ast_histories) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize graph");
    println!("  - 最適化完了");
    println!();

    // 生成されたOpenCLコードを表示
    println!("=== 生成されたOpenCLコード ===");
    let mut opencl_renderer = OpenCLRenderer::new();
    let code = opencl_renderer.render_program(&optimized_program);
    println!("{}", code);

    // 最適化の統計情報を表示
    print_optimization_stats(&pipeline, &ast_histories);

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

    println!();

    // 可視化アプリケーションを起動
    println!("可視化UIを起動中...");
    println!("  - Graph Viewerタブ: グラフ最適化の履歴を表示");
    println!("  - Code Viewerタブ: AST最適化の履歴と最終コードを表示");
    println!("  - 矢印キーで前後のステップに移動できます");
    println!();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("Harp Pipeline Optimization Visualizer - RuntimeSelector Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Pipeline Visualizer",
        options,
        Box::new(move |_cc| {
            // OpenCLレンダラータイプでHarpVizAppを作成
            let mut app = HarpVizApp::with_renderer_type(RendererType::OpenCL);

            // グラフ最適化履歴を読み込む
            if let Some(graph_history) = pipeline.histories.combined_graph_history() {
                app.load_graph_optimization_history(graph_history);
            }

            // AST最適化履歴を読み込む（RuntimeSelectorを使った実測値ベースの最適化履歴）
            if let Some(history) = ast_histories.get("program") {
                app.load_ast_optimization_history(history.clone());
            }

            // AST最適化済みのProgramをCode Viewerに直接ロード
            app.load_optimized_ast(optimized_program);

            Ok(Box::new(app))
        }),
    )
}

/// 行列積を計算するヘルパー関数
/// C = A @ B (A: [M, K], B: [K, N]) -> C: [M, N]
fn matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    use harp::graph::ops::fused_elementwise_reduce;

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

    // B: [K, N] -> transpose -> [N, K]
    let b_transposed = b.view(b.view.clone().permute(vec![1, 0]));

    // A: [M, K] -> [M, 1, K]
    // B_T: [N, K] -> [1, N, K]
    let a_unsqueezed = a.view(a.view.clone().unsqueeze(1));
    let b_t_unsqueezed = b_transposed.view(b_transposed.view.clone().unsqueeze(0));

    let expanded_shape = vec![m, n, k_a];
    let a_expanded = a_unsqueezed.view(a_unsqueezed.view.clone().expand(expanded_shape.clone()));
    let b_t_expanded = b_t_unsqueezed.view(b_t_unsqueezed.view.clone().expand(expanded_shape));

    // FusedElementwiseReduceを使用
    let expr = wildcard("0") * wildcard("1");
    fused_elementwise_reduce(vec![a_expanded, b_t_expanded], expr, ReduceOp::Sum, 2)
}

/// 複雑な計算グラフを作成（行列積と定数演算を含む）
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

/// 最適化の統計情報を表示
fn print_optimization_stats(
    pipeline: &GenericPipeline<OpenCLRenderer, OpenCLCompiler>,
    ast_histories: &std::collections::HashMap<String, harp::opt::ast::OptimizationHistory>,
) {
    println!("=== 最適化統計 ===");

    // グラフ最適化
    if let Some(graph_history) = &pipeline.histories.graph {
        println!("\n[グラフ最適化]");
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
    }

    // AST最適化（RuntimeSelector使用）
    if let Some(ast_history) = ast_histories.get("program") {
        println!("\n[AST最適化 (RuntimeSelector)]");
        println!("  ステップ数: {}", ast_history.len());

        if let (Some(first), Some(last)) =
            (ast_history.get(0), ast_history.get(ast_history.len() - 1))
        {
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
