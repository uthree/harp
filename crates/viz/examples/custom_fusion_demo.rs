//! CustomFusionSuggesterによる段階的融合の可視化デモ
//!
//! 連続するElementwise演算がGraphOp::Customに段階的に融合される様子を
//! 可視化します。

use harp::graph::{DType, Graph};
use harp::opt::graph::{BeamSearchGraphOptimizer, FusionSuggester, SimpleCostEstimator};
use harp_viz::HarpVizApp;

fn main() -> eframe::Result {
    // ログを初期化
    harp::opt::log_capture::init_with_env_logger();

    println!("=== CustomFusionSuggester 段階的融合デモ ===\n");
    println!("このデモでは、連続するElementwise演算が");
    println!("GraphOp::Customに段階的に融合される過程を可視化します。\n");

    println!("【計算グラフの構造】");
    println!("  入力: a, b, c, d, e (各 [100] のF32テンソル)");
    println!("  演算: ((((a + b) * c) - d) / e)");
    println!("  期待される融合:");
    println!("    Step 0: 初期状態 (5つのElementwise演算)");
    println!("    Step 1: (a + b) と * c を融合");
    println!("    Step 2: 結果と - d を融合");
    println!("    Step 3: 結果と / e を融合");
    println!("    最終: 1つのCustomノード\n");

    // グラフを作成
    let graph = create_elementwise_chain_graph();
    println!(
        "グラフ作成完了: {} 入力, {} 出力\n",
        graph.input_metas().len(),
        graph.outputs().len()
    );

    // CustomFusionSuggesterのみを使って最適化
    println!("【最適化実行中...】");
    let suggester = FusionSuggester::new();
    let estimator = SimpleCostEstimator::new();

    let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
        .with_beam_width(1) // ビーム幅1で1つずつ融合
        .with_max_steps(10)
        .with_progress(true)
        .with_collect_logs(true);

    let (optimized_graph, history) = optimizer.optimize_with_history(graph);

    println!("\n最適化完了!");
    println!("  ステップ数: {}", history.len());
    println!(
        "  初期コスト: {:.2}",
        history.get(0).map(|s| s.cost).unwrap_or(0.0)
    );
    println!(
        "  最終コスト: {:.2}",
        history
            .get(history.len() - 1)
            .map(|s| s.cost)
            .unwrap_or(0.0)
    );

    // 各ステップの説明を表示
    println!("\n【各ステップの詳細】");
    for (i, snapshot) in history.snapshots().iter().enumerate() {
        let outputs = snapshot.graph.outputs();
        let output = outputs.get("result").unwrap();
        let op_name = format_op_type(&output.op);
        let num_inputs = output.src.len();
        println!("  Step {}: {} (入力数: {})", i, op_name, num_inputs);
    }

    // 最終グラフの出力ノードを確認
    println!("\n【最終グラフの構造】");
    let final_outputs = optimized_graph.outputs();
    let final_output = final_outputs.get("result").unwrap();
    println!("  出力ノード: {}", format_op_type(&final_output.op));
    println!("  入力数: {}", final_output.src.len());

    // 可視化を起動
    println!("\n可視化UIを起動中...");
    println!("  - ◀ Prev / Next ▶ ボタンでステップを移動");
    println!("  - 各ステップでノードが融合される様子を確認できます\n");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("CustomFusionSuggester Demo - Staged Fusion Visualization"),
        ..Default::default()
    };

    eframe::run_native(
        "CustomFusion Demo",
        options,
        Box::new(move |_cc| {
            let mut app = HarpVizApp::new();
            app.load_graph_optimization_history(history);
            Ok(Box::new(app))
        }),
    )
}

/// Elementwise演算のチェーンを含むグラフを作成
/// ((((a + b) * c) - d) / e)
fn create_elementwise_chain_graph() -> Graph {
    let mut graph = Graph::new();

    // 入力テンソル
    let a = graph.input("a", DType::F32, vec![100]);
    let b = graph.input("b", DType::F32, vec![100]);
    let c = graph.input("c", DType::F32, vec![100]);
    let d = graph.input("d", DType::F32, vec![100]);
    let e = graph.input("e", DType::F32, vec![100]);

    // Elementwise演算のチェーン
    let t1 = a + b; // Add
    let t2 = t1 * c; // Mul
    let t3 = t2 - d; // Sub (Add + Neg)
    let result = t3 / e; // Div (Mul + Recip)

    graph.output("result", result);
    graph
}

/// GraphOpを読みやすい形式で表示
fn format_op_type(op: &harp::graph::GraphOp) -> String {
    use harp::graph::GraphOp;
    match op {
        GraphOp::Buffer { name } => name.clone(),
        GraphOp::Const(_) => "Const".to_string(),
        GraphOp::Elementwise { op, .. } => format!("Elementwise({:?})", op),
        GraphOp::Custom { ast, .. } => {
            format!("Custom(ast={})", format_ast_brief(ast))
        }
        GraphOp::FusedElementwise { expr, .. } => {
            format!("FusedElementwise({})", format_ast_brief(expr))
        }
        _ => format!("{:?}", op),
    }
}

/// ASTノードを簡潔に表示
fn format_ast_brief(ast: &harp::ast::AstNode) -> String {
    use harp::ast::AstNode;
    match ast {
        AstNode::Wildcard(name) => format!("W({})", name),
        AstNode::Add(l, r) => format!("({} + {})", format_ast_brief(l), format_ast_brief(r)),
        AstNode::Mul(l, r) => format!("({} * {})", format_ast_brief(l), format_ast_brief(r)),
        AstNode::Recip(x) => format!("1/{}", format_ast_brief(x)),
        AstNode::Const(lit) => format!("{:?}", lit),
        _ => "...".to_string(),
    }
}
