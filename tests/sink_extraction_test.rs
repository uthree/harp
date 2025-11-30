/// SinkノードのProgram抽出テスト
///
/// SinkAbsorptionSuggesterで生成されたProgramが正しく抽出されることを確認

#[test]
fn test_sink_program_extraction() {
    use harp::backend::GenericPipeline;
    use harp::backend::c::{CCompiler, CRenderer};
    use harp::graph::{DType, Graph};

    // シンプルなグラフ
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);
    let c = &a + &b;
    graph.output("c", c);

    eprintln!("Initial graph sink: {:?}", graph.sink().is_some());

    // パイプラインで最適化
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);
    pipeline.enable_ast_optimization = false;

    let (program, _) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize");

    // 生成されたProgramを確認
    if let harp::ast::AstNode::Program { functions, .. } = &program {
        eprintln!("Program has {} functions", functions.len());
        for func in functions {
            match func {
                harp::ast::AstNode::Kernel { name, .. } => {
                    eprintln!("  Kernel: {:?}", name);
                }
                harp::ast::AstNode::Function { name, .. } => {
                    eprintln!("  Function: {:?}", name);
                }
                _ => {}
            }
        }

        // カーネルとmain関数があることを確認
        assert!(functions.len() >= 2, "Should have at least kernel and main");

        // harp_mainが存在することを確認
        let has_main = functions.iter().any(
            |f| matches!(f, harp::ast::AstNode::Function { name: Some(n), .. } if n == "harp_main"),
        );
        assert!(has_main, "Should have harp_main function");
    } else {
        panic!("Expected Program node");
    }
}

#[test]
fn test_sink_program_extraction_complex() {
    use harp::backend::GenericPipeline;
    use harp::backend::c::{CCompiler, CRenderer};
    use harp::graph::{DType, Graph};

    // より複雑なグラフ: reduce(a + b) + c
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 5]);
    let b = graph.input("b", DType::F32, vec![10, 5]);
    let c = graph.input("c", DType::F32, vec![10]);
    let sum = &a + &b;
    let reduced = sum.reduce_sum(1);
    let result = &reduced + &c;
    graph.output("result", result);

    eprintln!("Initial graph sink: {:?}", graph.sink().is_some());

    // パイプラインで最適化
    let renderer = CRenderer::new();
    let compiler = CCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);
    pipeline.enable_ast_optimization = false;

    let (program, _) = pipeline
        .optimize_graph_with_all_histories(graph)
        .expect("Failed to optimize");

    // 生成されたProgramを確認
    if let harp::ast::AstNode::Program { functions, .. } = &program {
        eprintln!("Program has {} functions", functions.len());
        for func in functions {
            match func {
                harp::ast::AstNode::Kernel { name, .. } => {
                    eprintln!("  Kernel: {:?}", name);
                }
                harp::ast::AstNode::Function { name, .. } => {
                    eprintln!("  Function: {:?}", name);
                }
                _ => {}
            }
        }

        // カーネルとmain関数があることを確認
        // 複雑なグラフでは3つ以上の関数（複数カーネル + main）
        assert!(
            functions.len() >= 2,
            "Should have at least kernels and main"
        );

        // harp_mainが存在することを確認
        let has_main = functions.iter().any(
            |f| matches!(f, harp::ast::AstNode::Function { name: Some(n), .. } if n == "harp_main"),
        );
        assert!(has_main, "Should have harp_main function");
    } else {
        panic!("Expected Program node");
    }
}
