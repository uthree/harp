use crate::ast::AstNode;
use crate::ast::pat::{AstRewriteRule, AstRewriter};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use std::rc::Rc;

use super::estimator::CostEstimator;
use super::history::{OptimizationHistory, OptimizationSnapshot};
use super::suggester::Suggester;

/// ASTを最適化するトレイト
pub trait Optimizer {
    /// ASTを最適化して返す
    fn optimize(&self, ast: AstNode) -> AstNode;
}

/// ルールベースの最適化器
pub struct RuleBaseOptimizer {
    rewriter: AstRewriter,
}

impl RuleBaseOptimizer {
    /// 新しい最適化器を作成
    pub fn new(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self {
            rewriter: AstRewriter::new(rules),
        }
    }

    /// 最大反復回数を設定
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.rewriter = self.rewriter.with_max_iterations(max);
        self
    }
}

impl Optimizer for RuleBaseOptimizer {
    fn optimize(&self, ast: AstNode) -> AstNode {
        debug!("RuleBaseOptimizer: Starting optimization");
        let result = self.rewriter.apply(ast);
        debug!("RuleBaseOptimizer: Optimization complete");
        result
    }
}

/// ビームサーチ最適化器
///
/// # 終了条件
///
/// 最適化は以下のいずれかの条件で終了します：
/// - 最大ステップ数(`max_steps`)に達した
/// - Suggesterから新しい提案がなくなった（これ以上最適化できない）
pub struct BeamSearchOptimizer<S, E>
where
    S: Suggester,
    E: CostEstimator,
{
    suggester: S,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
}

impl<S, E> BeamSearchOptimizer<S, E>
where
    S: Suggester,
    E: CostEstimator,
{
    /// 新しいビームサーチ最適化器を作成
    pub fn new(suggester: S, estimator: E) -> Self {
        Self {
            suggester,
            estimator,
            beam_width: 10,
            max_steps: 10000,
            show_progress: true,
        }
    }

    /// ビーム幅を設定
    ///
    /// 各ステップで保持する候補の最大数
    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    /// 最大ステップ数を設定
    ///
    /// 最適化の最大反復回数。この回数に達するか、Suggesterからの提案がなくなると終了します。
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// 最大深さを設定（`with_max_steps`のエイリアス）
    #[deprecated(since = "0.1.0", note = "Use `with_max_steps` instead")]
    pub fn with_max_depth(self, depth: usize) -> Self {
        self.with_max_steps(depth)
    }

    /// プログレスバーの表示/非表示を設定
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }
}

impl<S, E> BeamSearchOptimizer<S, E>
where
    S: Suggester,
    E: CostEstimator,
{
    /// 履歴を記録しながら最適化を実行
    pub fn optimize_with_history(&self, ast: AstNode) -> (AstNode, OptimizationHistory) {
        use crate::opt::log_capture;

        // ログキャプチャを開始
        log_capture::start_capture();

        debug!("BeamSearchOptimizer: Starting beam search optimization with history");

        let mut history = OptimizationHistory::new();
        let mut beam = vec![ast.clone()];

        // 初期状態を記録
        let initial_cost = self.estimator.estimate(&ast);
        let initial_logs = log_capture::get_captured_logs();
        history.add_snapshot(OptimizationSnapshot::with_logs(
            0,
            ast,
            initial_cost,
            "Initial AST".to_string(),
            0,
            None,
            initial_logs,
        ));

        // 初期ログをクリア（各ステップで新しいログのみを記録するため）
        log_capture::clear_logs();

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_steps as u64);
            pb.set_style(
                ProgressStyle::with_template("{prefix:>12.cyan.bold} [{bar:24}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=> "),
            );
            pb.set_prefix("Optimizing");
            Some(pb)
        } else {
            None
        };

        let mut actual_steps = 0;
        for step in 0..self.max_steps {
            actual_steps = step;
            if let Some(ref pb) = pb {
                pb.set_message(format!("step {}", step + 1));
                pb.set_position(step as u64);
            }

            let mut candidates = Vec::new();

            // 現在のビーム内の各候補から新しい候補を生成
            for ast in &beam {
                let new_candidates = self.suggester.suggest(ast);
                candidates.extend(new_candidates);
            }

            if candidates.is_empty() {
                debug!(
                    "BeamSearchOptimizer: No more candidates at step {} - optimization complete (early termination)",
                    step
                );
                // 早期終了時は実際のステップ数を記録
                actual_steps = step;
                if let Some(ref pb) = pb {
                    pb.set_position(step as u64);
                    pb.set_message(format!("converged at step {}", step));
                }
                break;
            }

            debug!(
                "BeamSearchOptimizer: Found {} candidates at step {}",
                candidates.len(),
                step
            );

            // コストでソートして上位beam_width個を残す
            candidates.sort_by(|a, b| {
                self.estimator
                    .estimate(a)
                    .partial_cmp(&self.estimator.estimate(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            beam = candidates.into_iter().take(self.beam_width).collect();

            // このステップの最良候補を記録
            if let Some(best) = beam.first() {
                let cost = self.estimator.estimate(best);
                let step_logs = log_capture::get_captured_logs();
                history.add_snapshot(OptimizationSnapshot::with_logs(
                    step + 1,
                    best.clone(),
                    cost,
                    format!("Step {}: beam width {}", step + 1, beam.len()),
                    0,
                    None,
                    step_logs,
                ));
                // このステップのログをクリア（次のステップで新しいログのみを記録するため）
                log_capture::clear_logs();
            }
        }

        if let Some(pb) = pb {
            pb.finish_and_clear();
            // Cargoスタイルの完了メッセージ
            if actual_steps + 1 < self.max_steps {
                // 早期終了
                println!(
                    "{:>12} AST optimization (converged after {} steps)",
                    "\x1b[1;32mFinished\x1b[0m",
                    actual_steps + 1
                );
            } else {
                // 最大ステップ数に到達
                println!(
                    "{:>12} AST optimization ({} steps)",
                    "\x1b[1;32mFinished\x1b[0m",
                    actual_steps + 1
                );
            }
        }

        debug!("BeamSearchOptimizer: Beam search optimization complete");

        // 最良の候補を返す
        let best = beam
            .into_iter()
            .min_by(|a, b| {
                self.estimator
                    .estimate(a)
                    .partial_cmp(&self.estimator.estimate(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        (best, history)
    }
}

impl<S, E> Optimizer for BeamSearchOptimizer<S, E>
where
    S: Suggester,
    E: CostEstimator,
{
    fn optimize(&self, ast: AstNode) -> AstNode {
        let (optimized, _) = self.optimize_with_history(ast);
        optimized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;
    use crate::astpat;
    use crate::opt::ast::estimator::SimpleCostEstimator;
    use crate::opt::ast::suggester::RuleBaseSuggester;

    #[test]
    fn test_rule_base_optimizer() {
        // Add(a, 0) -> a というルール
        let rule = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Int(0))))
        } => {
            a
        });

        let optimizer = RuleBaseOptimizer::new(vec![rule]);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(42))),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Int(42)));
    }

    #[test]
    fn test_beam_search_optimizer() {
        // 交換則と単位元除去のルール
        let rule1 = astpat!(|a, b| {
            AstNode::Add(Box::new(a), Box::new(b))
        } => {
            AstNode::Add(Box::new(b), Box::new(a))
        });

        let rule2 = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Int(0))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule1, rule2]);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(5)
            .with_max_steps(5)
            .with_progress(false); // テスト中はプログレスバーを非表示

        // (42 + 0) を最適化
        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(42))),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input);
        // 最終的に42に簡約されるはず
        assert_eq!(result, AstNode::Const(Literal::Int(42)));
    }

    #[test]
    fn test_beam_search_optimizer_complex() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(10)
            .with_max_steps(10)
            .with_progress(false);

        // ((2 + 3) * 1) + 0 を最適化
        let input = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Int(2))),
                    Box::new(AstNode::Const(Literal::Int(3))),
                )),
                Box::new(AstNode::Const(Literal::Int(1))),
            )),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input);
        // 最終的に5に簡約されるはず
        assert_eq!(result, AstNode::Const(Literal::Int(5)));
    }

    #[test]
    fn test_beam_search_no_applicable_rules() {
        // マッチしないルールのみ
        let rule = astpat!(|a| {
            AstNode::Mul(Box::new(a), Box::new(AstNode::Const(Literal::Int(99))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule]);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(5)
            .with_max_steps(5)
            .with_progress(false);

        // ルールが適用されない入力
        let input = AstNode::Const(Literal::Int(42));
        let result = optimizer.optimize(input.clone());

        // 変更されないはず
        assert_eq!(result, input);
    }

    #[test]
    fn test_beam_search_already_optimal() {
        use crate::opt::ast::rules::all_algebraic_rules;

        let suggester = RuleBaseSuggester::new(all_algebraic_rules());
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(10)
            .with_max_steps(10)
            .with_progress(false);

        // すでに最適化済みの入力
        let input = AstNode::Const(Literal::Int(42));
        let result = optimizer.optimize(input.clone());

        // 変更されないはず
        assert_eq!(result, input);
    }

    #[test]
    fn test_beam_search_with_beam_width_one() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);
        let estimator = SimpleCostEstimator::new();

        // ビーム幅1（貪欲法）
        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(1)
            .with_max_steps(10)
            .with_progress(false);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(5))),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input);
        // ビーム幅1でも最適化できるはず
        assert_eq!(result, AstNode::Const(Literal::Int(5)));
    }

    #[test]
    fn test_beam_search_with_max_depth_zero() {
        use crate::opt::ast::rules::all_algebraic_rules;

        let suggester = RuleBaseSuggester::new(all_algebraic_rules());
        let estimator = SimpleCostEstimator::new();

        // 最大ステップ数0（最適化しない）
        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(10)
            .with_max_steps(0)
            .with_progress(false);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(5))),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input.clone());
        // 変更されないはず
        assert_eq!(result, input);
    }

    #[test]
    fn test_beam_search_early_termination() {
        // 1回で最適化が完了し、それ以降候補がなくなるケース
        let rule = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Int(0))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule]);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(5)
            .with_max_steps(10) // 最大ステップ数は10だが早期終了するはず
            .with_progress(false);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(42))),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Int(42)));
    }

    #[test]
    fn test_beam_search_large_beam_width() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);
        let estimator = SimpleCostEstimator::new();

        // 非常に大きなビーム幅
        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(1000)
            .with_max_steps(5)
            .with_progress(false);

        let input = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Int(2))),
                    Box::new(AstNode::Const(Literal::Int(3))),
                )),
                Box::new(AstNode::Const(Literal::Int(1))),
            )),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Int(5)));
    }
}
