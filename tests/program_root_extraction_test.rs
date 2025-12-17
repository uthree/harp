/// Program抽出テスト
///
/// 最適化後のグラフからProgramが正しく抽出されることを確認
/// Note: ProgramRootAbsorptionは削除されたため、各KernelがProgramの関数として出力される

#[test]
fn test_program_root_extraction() {
    use harp::backend::{MultiPhaseConfig, create_multi_phase_optimizer};
    use harp::graph::{DType, Graph};
    use harp::lowerer::extract_program;
    use harp::opt::graph::GraphOptimizer;

    // シンプルなグラフ
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);
    let c = &a + &b;
    graph.output("c", c);

    eprintln!("Initial graph outputs: {}", graph.outputs().len());

    // グラフ最適化
    let config = MultiPhaseConfig::new()
        .with_beam_width(4)
        .with_max_steps(1000)
        .with_progress(false)
        .with_collect_logs(false);

    let optimizer = create_multi_phase_optimizer(config);
    let (optimized_graph, _) = optimizer.optimize_with_history(graph);

    // ASTに変換
    let program = extract_program(optimized_graph);

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

        // 少なくとも1つの関数があることを確認
        assert!(!functions.is_empty(), "Should have at least one function");

        // Kernel または Function が存在することを確認
        // シンプルな演算はFunctionとして、複雑な演算はKernelとして生成される
        let has_callable = functions.iter().any(|f| {
            matches!(
                f,
                harp::ast::AstNode::Kernel { .. } | harp::ast::AstNode::Function { .. }
            )
        });
        assert!(has_callable, "Should have at least one callable function");
    } else {
        panic!("Expected Program node");
    }
}

#[test]
fn test_program_root_extraction_complex() {
    use harp::backend::{MultiPhaseConfig, create_multi_phase_optimizer};
    use harp::graph::{DType, Graph};
    use harp::lowerer::extract_program;
    use harp::opt::graph::GraphOptimizer;

    // より複雑なグラフ: reduce(a + b) + c
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 5]);
    let b = graph.input("b", DType::F32, vec![10, 5]);
    let c = graph.input("c", DType::F32, vec![10]);
    let sum = &a + &b;
    let reduced = sum.reduce_sum(1);
    let result = &reduced + &c;
    graph.output("result", result);

    eprintln!("Initial graph outputs: {}", graph.outputs().len());

    // グラフ最適化
    let config = MultiPhaseConfig::new()
        .with_beam_width(4)
        .with_max_steps(1000)
        .with_progress(false)
        .with_collect_logs(false);

    let optimizer = create_multi_phase_optimizer(config);
    let (optimized_graph, _) = optimizer.optimize_with_history(graph);

    // ASTに変換
    let program = extract_program(optimized_graph);

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

        // 少なくとも1つの関数があることを確認
        assert!(!functions.is_empty(), "Should have at least one function");

        // Kernel または Function が存在することを確認
        // シンプルな演算はFunctionとして、複雑な演算はKernelとして生成される
        let has_callable = functions.iter().any(|f| {
            matches!(
                f,
                harp::ast::AstNode::Kernel { .. } | harp::ast::AstNode::Function { .. }
            )
        });
        assert!(has_callable, "Should have at least one callable function");
    } else {
        panic!("Expected Program node");
    }
}
