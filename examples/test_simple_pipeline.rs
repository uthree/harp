use harp::backend::openmp::{CCompiler, CRenderer};
use harp::backend::{GenericPipeline, OptimizationConfig};
use harp::graph::{DType, Graph};

fn main() {
    harp::opt::log_capture::init_with_env_logger();

    let renderer = CRenderer::new();
    let compiler = CCompiler::new();

    let mut pipeline = GenericPipeline::new(renderer, compiler);
    pipeline.enable_ast_optimization = true;
    pipeline.ast_config = OptimizationConfig {
        beam_width: 4,
        max_steps: 100,
        show_progress: false,
    };

    // シンプルなグラフ: a + b
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![256, 64])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![256, 64])
        .build();
    let result = a + b;
    graph.output("result", result);

    println!("=== 最適化実行 ===");
    let (program, _) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize");

    println!("\n=== 生成されたコード ===");
    use harp::ast::renderer::render_ast;
    let code = render_ast(&program);
    println!("{}", code);

    // Assign文の数を確認
    let assign_count = code.matches("alu").count();
    println!("\n'alu'の出現回数: {}", assign_count);

    if code.contains("= alu") && !code.contains("alu0 =") {
        eprintln!("\n❌ 警告: alu変数が定義されずに使用されています！");
    }
}
