/// Vectorization test
/// ベクトル化戦略が実際にコード生成に適用されることを確認します。
use harp::ast::DType;
use harp::backend::c::renderer::CRenderer;
use harp::backend::Renderer;
use harp::graph::{Graph, LoopStrategy};
use harp::lowerer::Lowerer;

fn main() {
    println!("=== Vectorization Test ===\n");

    // 1. グラフを作成
    let mut graph = Graph::new();

    // 入力: [128] の F32 配列 2つ
    let a = graph.input(DType::F32, vec![128.into()]);
    let b = graph.input(DType::F32, vec![128.into()]);

    // 要素ごとの演算: c = a + b
    let c = a + b;

    println!("Test 1: Without vectorization strategy");
    println!("  Creating graph without strategy...");
    graph.output(c.clone());

    let mut lowerer = Lowerer::new();
    let ast_without_vec = lowerer.lower(&graph);

    // C コードにレンダリング
    let mut renderer = CRenderer::new();
    let c_code_without_vec = renderer.render(ast_without_vec);

    println!("  Generated C code (first 500 chars):");
    println!(
        "  {}",
        &c_code_without_vec.chars().take(500).collect::<String>()
    );
    println!();

    // 2. ベクトル化戦略を手動で適用
    println!("Test 2: With vectorization strategy");
    println!("  Applying vectorization strategy (axis=0, width=8)...");

    let mut graph_with_vec = Graph::new();
    let a2 = graph_with_vec.input(DType::F32, vec![128.into()]);
    let b2 = graph_with_vec.input(DType::F32, vec![128.into()]);
    let c2 = a2 + b2;

    // LoopStrategy を手動で設定
    let strategy = LoopStrategy {
        vectorize: Some((0, 8)),
        unroll: None,
        parallelize: vec![],
        tile: vec![],
        use_shared_memory: false,
    };

    let c2_with_strategy = c2.with_strategy(strategy);
    graph_with_vec.output(c2_with_strategy);

    let mut lowerer2 = Lowerer::new();
    let ast_with_vec = lowerer2.lower(&graph_with_vec);

    let mut renderer2 = CRenderer::new();
    let c_code_with_vec = renderer2.render(ast_with_vec);

    println!("  Generated C code (first 500 chars):");
    println!(
        "  {}",
        &c_code_with_vec.chars().take(500).collect::<String>()
    );
    println!();

    // 3. 違いを確認
    println!("Test 3: Checking differences");
    let has_vector_load =
        c_code_with_vec.contains("vector_width") || c_code_with_vec != c_code_without_vec;

    if has_vector_load {
        println!("  ✅ Vectorization strategy was applied!");
        println!(
            "  Code length without vec: {} chars",
            c_code_without_vec.len()
        );
        println!("  Code length with vec:    {} chars", c_code_with_vec.len());
    } else {
        println!("  ⚠️  Vectorization strategy was NOT reflected in generated code");
    }
    println!();

    println!("=== Test completed ===");
}
