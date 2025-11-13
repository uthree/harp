use harp::ast::helper::{assign, load, store, var};
use harp::ast::renderer::render_ast_with;
use harp::ast::{AstNode, DType, Literal, Mutability, Scope};
use harp::backend::openmp::CRenderer;
use harp::opt::ast::estimator::SimpleCostEstimator;
use harp::opt::ast::optimizer::BeamSearchOptimizer;
use harp::opt::ast::rules::all_algebraic_rules;
use harp::opt::ast::suggesters::RuleBaseSuggester;
use harp::opt::ast::Optimizer;

fn main() {
    let input0_ptr = var("input0");
    let ridx0 = var("ridx0");
    let ridx1 = var("ridx1");

    // オフセット計算
    let offset = (ridx0.clone() * AstNode::Const(Literal::Int(64))) + ridx1.clone();

    // alu0 = input0[offset]
    let alu0_value = load(input0_ptr.clone(), offset.clone(), DType::F32);
    let stmt1 = assign("alu0", alu0_value);

    // output[offset] = alu0
    let output_ptr = var("output");
    let stmt2 = store(output_ptr, offset.clone(), var("alu0"));

    let mut scope = Scope::new();
    scope
        .declare("alu0".to_string(), DType::F32, Mutability::Mutable)
        .unwrap();

    let block = AstNode::Block {
        statements: vec![stmt1, stmt2],
        scope: Box::new(scope),
    };

    println!("=== 最適化前 ===");
    let renderer = CRenderer::new();
    let code_before = render_ast_with(&block, &renderer);
    println!("{}", code_before);

    // 最適化実行
    let rules = all_algebraic_rules();
    let suggester = RuleBaseSuggester::new(rules);
    let estimator = SimpleCostEstimator::new();

    let optimizer = BeamSearchOptimizer::new(suggester, estimator)
        .with_beam_width(10)
        .with_max_steps(100)
        .with_progress(false);

    let optimized = optimizer.optimize(block);

    println!("\n=== 最適化後 ===");
    let code_after = render_ast_with(&optimized, &renderer);
    println!("{}", code_after);

    // Assign文が保持されているか確認
    if let AstNode::Block { statements, .. } = &optimized {
        let assign_count = statements
            .iter()
            .filter(|s| matches!(s, AstNode::Assign { .. }))
            .count();
        println!(
            "\n✓ Assign文の数: {} (期待値: 1)",
            assign_count
        );

        if assign_count == 0 {
            eprintln!("❌ エラー: Assign文が消えています！");
        }
    }
}
