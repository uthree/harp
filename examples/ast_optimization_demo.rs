//! AST最適化のデモ
//!
//! ビームサーチによる最適化の過程を可視化します

use harp::ast::helper::*;
use harp::ast::renderer::render_ast;
use harp::ast::{AstNode, Literal};
use harp::opt::ast::{BeamSearchOptimizer, OptimizationHistory, RuleBaseSuggester};

fn main() {
    env_logger::init();

    println!("AST最適化のデモ\n");

    // 最適化前のAST: ((a + 0) * 1) + (b * 0)
    let ast = create_sample_ast();

    println!("【最適化前のAST】");
    println!("{}\n", render_ast(&ast));

    // ルールベースの提案器を作成
    let rules = harp::opt::ast::rules::all_algebraic_rules();
    let suggester = RuleBaseSuggester::new(rules);

    // ビームサーチ最適化器を作成（SelectorがCostEstimatorを内包）
    let optimizer = BeamSearchOptimizer::new(suggester)
        .with_beam_width(5)
        .with_max_steps(5)
        .with_progress(true);

    // 最適化を実行（履歴付き）
    println!("\n【最適化実行中】");
    let (optimized_ast, history) = optimizer.optimize_with_history(ast);

    println!("\n【最適化後のAST】");
    println!("{}\n", render_ast(&optimized_ast));

    // 履歴を表示
    display_history(&history);

    // コスト遷移を表示
    display_cost_transition(&history);
}

fn create_sample_ast() -> AstNode {
    // ((a + 0) * 1) + (b * 0)
    let a = var("a");
    let b = var("b");
    let zero = AstNode::Const(Literal::Int(0));
    let one = AstNode::Const(Literal::Int(1));

    let a_plus_zero = a + zero.clone();
    let left = a_plus_zero * one;

    let b_times_zero = b * zero;

    left + b_times_zero
}

fn display_history(history: &OptimizationHistory) {
    println!("\n【最適化履歴】");
    println!("{} ステップの履歴が記録されました\n", history.len());

    // 各ステップの最良候補を表示
    let mut current_step = 0;
    for snapshot in history.snapshots() {
        if snapshot.step != current_step {
            current_step = snapshot.step;
            println!("--- Step {} ---", current_step);
        }

        if snapshot.rank == 0 {
            // 最良候補のみ表示
            println!("  Rank {}: Cost = {:.2}", snapshot.rank, snapshot.cost);
            println!("  AST: {}", render_ast(&snapshot.ast));
            println!();
        }
    }
}

fn display_cost_transition(history: &OptimizationHistory) {
    println!("\n【コスト遷移】");
    let transition = history.cost_transition();

    for (step, cost) in &transition {
        let bar_length = (100.0 - cost).clamp(0.0, 50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("Step {}: {:.2} {}", step, cost, bar);
    }

    if let (Some((_, initial_cost)), Some((_, final_cost))) =
        (transition.first(), transition.last())
    {
        let improvement = initial_cost - final_cost;
        let improvement_rate = (improvement / initial_cost) * 100.0;
        println!(
            "\n改善: {:.2} -> {:.2} ({:.1}% 削減)",
            initial_cost, final_cost, improvement_rate
        );
    }
}
