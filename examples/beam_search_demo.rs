use harp::ast::{AstNode, Literal};
use harp::opt::ast::rules::{add_commutative, all_algebraic_rules};
use harp::opt::ast::{
    AstCostEstimator, AstOptimizer, BeamSearchOptimizer, RuleBaseSuggester, SimpleCostEstimator,
};

fn main() {
    env_logger::init();

    println!("ビームサーチ最適化のデモ\n");

    // より複雑なAST式を作成（プログレスバーが見えるように多くの演算を含む）
    // ((a + b) + (c + d)) + ((e + f) + (g + h)) の形式で、各変数は (x + 0) * 1 のような冗長な式
    fn make_redundant(val: isize) -> AstNode {
        AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Int(val))),
                    Box::new(AstNode::Const(Literal::Int(0))),
                )),
                Box::new(AstNode::Const(Literal::Int(0))),
            )),
            Box::new(AstNode::Mul(
                Box::new(AstNode::Const(Literal::Int(1))),
                Box::new(AstNode::Const(Literal::Int(1))),
            )),
        )
    }

    let ast = AstNode::Add(
        Box::new(AstNode::Add(
            Box::new(AstNode::Add(
                Box::new(make_redundant(1)),
                Box::new(make_redundant(2)),
            )),
            Box::new(AstNode::Add(
                Box::new(make_redundant(3)),
                Box::new(make_redundant(4)),
            )),
        )),
        Box::new(AstNode::Add(
            Box::new(AstNode::Add(
                Box::new(make_redundant(5)),
                Box::new(make_redundant(6)),
            )),
            Box::new(AstNode::Add(
                Box::new(make_redundant(7)),
                Box::new(make_redundant(8)),
            )),
        )),
    );

    println!("元のAST: {:?}\n", ast);

    // ルールセットを作成（交換則と乗算の交換則も含む）
    let mut rules = all_algebraic_rules();
    rules.push(add_commutative());
    use harp::opt::ast::rules::mul_commutative;
    rules.push(mul_commutative());

    // ビームサーチ最適化器を作成（SelectorがCostEstimatorを内包）
    let suggester = RuleBaseSuggester::new(rules);

    let optimizer = BeamSearchOptimizer::new(suggester)
        .with_beam_width(20) // ビーム幅
        .with_max_steps(20) // 探索深さ（デバッグビルドでは各ステップ200ms待機するため約4秒）
        .with_progress(true); // プログレスバーを表示

    println!("ビームサーチ最適化を開始...\n");

    let optimized = optimizer.optimize(ast.clone());

    println!("\n最適化後のAST: {:?}", optimized);
    println!(
        "\n元のコスト: {}",
        SimpleCostEstimator::new().estimate(&ast)
    );
    println!(
        "最適化後のコスト: {}",
        SimpleCostEstimator::new().estimate(&optimized)
    );
}
