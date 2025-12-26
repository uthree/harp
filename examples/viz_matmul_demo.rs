//! 実際のMatMul最適化履歴可視化デモ（OpenCLバックエンド）
//!
//! 1024x1024の行列積を計算し、その最適化プロセスを可視化します。
//! OpenCLバックエンドを使用してGPU向けカーネルを生成します。
//!
//! # 実行方法
//! ```bash
//! cargo run --features "viz,opencl" --example viz_matmul_demo
//! ```
//!
//! # 操作方法
//! - ←/h: 前のステップへ
//! - →/l: 次のステップへ
//! - ↑/k: 前の候補を選択
//! - ↓/j: 次の候補を選択
//! - q/Esc: 終了

use harp::opt::ast::BeamSearchOptimizer;
use harp::opt::ast::rules::all_rules_with_search;
use harp::opt::ast::suggesters::{
    CompositeSuggester, FunctionInliningSuggester, GroupParallelizationSuggester,
    LocalParallelizationSuggester, LoopFusionSuggester, LoopInliningSuggester,
    LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseSuggester, VariableExpansionSuggester,
    VectorizationSuggester,
};
use harp::tensor::lowerer::TensorLowerer;
use harp::tensor::{Dim2, Tensor};

#[cfg(feature = "opencl")]
use harp::backend::Device;
#[cfg(feature = "opencl")]
use harp::backend::opencl::{OpenCLDevice, OpenCLRenderer};

fn main() {
    // ログを初期化（オプション）
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    // OpenCLデバイスの確認
    #[cfg(feature = "opencl")]
    {
        if !OpenCLDevice::is_available() {
            eprintln!("Error: OpenCL is not available on this system.");
            eprintln!("Please ensure you have OpenCL drivers installed.");
            std::process::exit(1);
        }

        match OpenCLDevice::new() {
            Ok(device) => {
                let profile = device.profile();
                println!("OpenCL Device detected:");
                println!("  Type: {:?}", profile.device_type);
                println!("  Compute units: {}", profile.compute_units);
                println!("  Max work group size: {}", profile.max_work_group_size);
                println!("  Local memory: {} KB", profile.local_memory_size / 1024);
                println!("  Warp/wavefront size: {}", profile.warp_size);
                println!();
            }
            Err(e) => {
                eprintln!("Error creating OpenCL device: {}", e);
                std::process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "opencl"))]
    {
        eprintln!("Error: This example requires the 'opencl' feature.");
        eprintln!("Run with: cargo run --features \"viz,opencl\" --example viz_matmul_demo");
        std::process::exit(1);
    }

    println!("Creating 1024x1024 matmul computation...");

    // 1024x1024の行列積を定義
    let size = 1024;
    let a = Tensor::<f32, Dim2>::input("A", [size, size]);
    let b = Tensor::<f32, Dim2>::input("B", [size, size]);
    let c = a.matmul2(&b);

    println!("Lowering to AST...");

    // TensorをASTに変換
    let mut lowerer = TensorLowerer::new();
    let ast = lowerer.lower(&c.into_dyn());

    println!("Setting up optimizer with GPU-oriented suggesters...");

    // GPU向け最適化用のSuggesterを構成
    let suggester = CompositeSuggester::new(vec![
        // ルールベース最適化（定数畳み込み、代数的簡約）
        Box::new(RuleBaseSuggester::new(all_rules_with_search())),
        // ループ最適化
        Box::new(LoopTilingSuggester::new()),
        Box::new(LoopInliningSuggester::new()),
        Box::new(LoopFusionSuggester::new()),
        Box::new(LoopInterchangeSuggester::new()),
        // GPU並列化
        Box::new(GroupParallelizationSuggester::new()),
        Box::new(LocalParallelizationSuggester::new()),
        // ベクトル化
        Box::new(VectorizationSuggester::new()),
        // 関数インライン化（最大100ノードまで）
        Box::new(FunctionInliningSuggester::new(100)),
        // 変数展開
        Box::new(VariableExpansionSuggester::new()),
    ]);

    // ビームサーチ最適化器を構成
    let mut optimizer = BeamSearchOptimizer::new(suggester)
        .with_beam_width(5)
        .with_max_steps(30)
        .with_collect_logs(true)
        .with_no_improvement_limit(Some(5))
        .without_progress(); // TUIと干渉しないようにプログレスを無効化

    println!("Running optimization...");

    // 最適化を実行して履歴を取得
    let (_optimized_ast, history) = optimizer.optimize_with_history(ast);

    println!("Optimization complete! {} steps recorded.", history.len());
    println!("Starting visualization...");
    println!("Press q or Esc to quit.");

    // 可視化を起動（OpenCLRendererを使用）
    #[cfg(all(feature = "viz", feature = "opencl"))]
    {
        let renderer = OpenCLRenderer::new();
        if let Err(e) = harp::viz::run_with_renderer(history, renderer) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    #[cfg(all(feature = "viz", not(feature = "opencl")))]
    {
        // OpenCLなしの場合はデフォルトレンダラーを使用
        if let Err(e) = harp::viz::run(history) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    #[cfg(not(feature = "viz"))]
    {
        let _ = history; // unused warning suppression
        eprintln!("Error: This example requires the 'viz' feature.");
        eprintln!("Run with: cargo run --features \"viz,opencl\" --example viz_matmul_demo");
        std::process::exit(1);
    }
}
