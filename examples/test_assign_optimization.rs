use harp::ast::helper::{assign, load, store, var};
use harp::ast::{AstNode, DType, Literal, Mutability, Scope};
use harp::opt::ast::Optimizer;
use harp::opt::ast::estimator::SimpleCostEstimator;
use harp::opt::ast::optimizer::BeamSearchOptimizer;
use harp::opt::ast::rules::all_algebraic_rules;
use harp::opt::ast::suggesters::RuleBaseSuggester;

fn main() {
    env_logger::init();

    // 問題を再現するテストケース
    // Block内のAssign文が最適化で消えるかテスト

    let input0_ptr = var("input0");
    let input1_ptr = var("input1");
    let output_ptr = var("output");
    let ridx0 = var("ridx0");
    let ridx1 = var("ridx1");

    // オフセット計算: (ridx0 * 64) + ridx1
    let offset = (ridx0.clone() * AstNode::Const(Literal::Int(64))) + ridx1.clone();

    // alu0 = input0[offset]
    let alu0_value = load(input0_ptr.clone(), offset.clone(), DType::F32);
    let stmt1 = assign("alu0", alu0_value);

    // alu1 = input1[offset]
    let alu1_value = load(input1_ptr.clone(), offset.clone(), DType::F32);
    let stmt2 = assign("alu1", alu1_value);

    // alu2 = alu0 + alu1
    let alu2_value = var("alu0") + var("alu1");
    let stmt3 = assign("alu2", alu2_value);

    // output[offset] = alu2
    let stmt4 = store(output_ptr, offset.clone(), var("alu2"));

    let mut scope = Scope::new();
    scope
        .declare("alu0".to_string(), DType::F32, Mutability::Mutable)
        .unwrap();
    scope
        .declare("alu1".to_string(), DType::F32, Mutability::Mutable)
        .unwrap();
    scope
        .declare("alu2".to_string(), DType::F32, Mutability::Mutable)
        .unwrap();

    let block = AstNode::Block {
        statements: vec![stmt1, stmt2, stmt3, stmt4],
        scope: Box::new(scope),
    };

    println!("=== 最適化前のBlock ===");
    println!(
        "文の数: {}",
        match &block {
            AstNode::Block { statements, .. } => statements.len(),
            _ => 0,
        }
    );

    // 最適化を実行
    let rules = all_algebraic_rules();
    let suggester = RuleBaseSuggester::new(rules);
    let estimator = SimpleCostEstimator::new();

    let optimizer = BeamSearchOptimizer::new(suggester, estimator)
        .with_beam_width(10)
        .with_max_steps(100)
        .with_progress(true);

    let optimized = optimizer.optimize(block);

    println!("\n=== 最適化後のBlock ===");

    // Assign文が保持されているか確認
    if let AstNode::Block { statements, .. } = &optimized {
        println!("文の数: {}", statements.len());
        for (i, stmt) in statements.iter().enumerate() {
            match stmt {
                AstNode::Assign { var, .. } => println!("  [{}] Assign to {}", i, var),
                AstNode::Store { .. } => println!("  [{}] Store", i),
                _ => println!("  [{}] Other", i),
            }
        }

        if statements.len() < 4 {
            eprintln!(
                "\n❌ エラー: Assign文が消えています！元は4文でしたが、{}文になっています",
                statements.len()
            );
            std::process::exit(1);
        } else {
            println!("\n✓ OK: すべての文が保持されています");
        }
    } else {
        eprintln!("\n❌ エラー: BlockがBlock以外のノードに変換されています！");
        std::process::exit(1);
    }
}
