use crate::ast::AstNode;
use crate::ast::pat::{AstRewriteRule, AstRewriter};
use crate::opt::selector::{Selector, StaticCostSelector};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, trace};
use std::rc::Rc;
use std::time::Instant;

use super::history::{OptimizationHistory, OptimizationSnapshot};
use super::{CostEstimator, Optimizer, Suggester};

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
        info!("AST rule-based optimization started");
        let result = self.rewriter.apply(ast);
        info!("AST rule-based optimization complete");
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
///
/// # 候補選択
///
/// `Selector`により候補選択処理を抽象化しています。
/// デフォルトは`StaticCostSelector`（静的コストでソートして上位n件を選択）。
/// `with_selector()`で二段階選択などのカスタム選択器を設定可能。
pub struct BeamSearchOptimizer<S, E, Sel = StaticCostSelector>
where
    S: Suggester,
    E: CostEstimator,
    Sel: Selector<AstNode>,
{
    suggester: S,
    estimator: E,
    selector: Sel,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
    collect_logs: bool,
    max_node_count: Option<usize>,
}

impl<S, E> BeamSearchOptimizer<S, E, StaticCostSelector>
where
    S: Suggester,
    E: CostEstimator,
{
    /// 新しいビームサーチ最適化器を作成
    ///
    /// デフォルトでは`StaticCostSelector`を使用します。
    pub fn new(suggester: S, estimator: E) -> Self {
        Self {
            suggester,
            estimator,
            selector: StaticCostSelector::new(),
            beam_width: 10,
            max_steps: 10000,
            show_progress: cfg!(debug_assertions),
            collect_logs: cfg!(debug_assertions),
            max_node_count: Some(10000), // デフォルトで最大1万ノードに制限
        }
    }
}

impl<S, E, Sel> BeamSearchOptimizer<S, E, Sel>
where
    S: Suggester,
    E: CostEstimator,
    Sel: Selector<AstNode>,
{
    /// カスタム選択器を設定
    ///
    /// デフォルトの`StaticCostSelector`の代わりに、
    /// `MultiStageSelector`などのカスタム選択器を使用できます。
    ///
    /// # Example
    ///
    /// ```ignore
    /// use harp::opt::{MultiStageSelector, BeamSearchOptimizer};
    ///
    /// let selector = MultiStageSelector::new()
    ///     .then(|ast| node_count(ast) as f32, 100)
    ///     .then(|ast| measure_cost(ast), 10);
    ///
    /// let optimizer = BeamSearchOptimizer::new(suggester, estimator)
    ///     .with_selector(selector);
    /// ```
    pub fn with_selector<NewSel>(self, selector: NewSel) -> BeamSearchOptimizer<S, E, NewSel>
    where
        NewSel: Selector<AstNode>,
    {
        BeamSearchOptimizer {
            suggester: self.suggester,
            estimator: self.estimator,
            selector,
            beam_width: self.beam_width,
            max_steps: self.max_steps,
            show_progress: self.show_progress,
            collect_logs: self.collect_logs,
            max_node_count: self.max_node_count,
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

    /// プログレスバーの表示/非表示を設定
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// ログ収集の有効/無効を設定
    pub fn with_collect_logs(mut self, collect: bool) -> Self {
        self.collect_logs = collect;
        self
    }

    /// 最大ノード数を設定
    ///
    /// ASTのノード数がこの値を超える候補は自動的に除外されます。
    /// Noneを指定すると制限なしになります。
    pub fn with_max_node_count(mut self, max: Option<usize>) -> Self {
        self.max_node_count = max;
        self
    }
}

impl<S, E, Sel> BeamSearchOptimizer<S, E, Sel>
where
    S: Suggester,
    E: CostEstimator,
    Sel: Selector<AstNode>,
{
    /// 履歴を記録しながら最適化を実行
    pub fn optimize_with_history(&self, ast: AstNode) -> (AstNode, OptimizationHistory) {
        use crate::opt::ast::estimator::SimpleCostEstimator;
        use crate::opt::log_capture;

        // 時間計測を開始
        let start_time = Instant::now();

        // ログキャプチャを開始（collect_logsが有効な場合のみ）
        if self.collect_logs {
            log_capture::start_capture();
        }

        info!(
            "AST beam search optimization started (beam_width={}, max_steps={}, max_nodes={:?})",
            self.beam_width, self.max_steps, self.max_node_count
        );

        let mut history = OptimizationHistory::new();
        let mut beam = vec![ast.clone()];

        // 初期状態を記録
        let initial_cost = self.estimator.estimate(&ast);
        info!("Initial AST cost: {:.2e}", initial_cost);
        let initial_logs = if self.collect_logs {
            log_capture::get_captured_logs()
        } else {
            Vec::new()
        };

        // これまでで最良の候補を保持（astを移動する前に初期化）
        let mut global_best = ast.clone();

        history.add_snapshot(OptimizationSnapshot::with_candidates(
            0,
            ast,
            initial_cost,
            "Initial AST".to_string(),
            0,
            None,
            initial_logs,
            0, // 初期状態では候補数は0
        ));

        // 初期ログをクリア（各ステップで新しいログのみを記録するため）
        if self.collect_logs {
            log_capture::clear_logs();
        }

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
        let mut best_cost = initial_cost;
        let mut no_improvement_count = 0;
        const MAX_NO_IMPROVEMENT_STEPS: usize = 3;

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

            // 最大ノード数制限を適用
            let original_count = candidates.len();
            if let Some(max_nodes) = self.max_node_count {
                candidates.retain(|ast| {
                    let node_count = SimpleCostEstimator::get_node_count(ast);
                    if node_count > max_nodes {
                        trace!(
                            "Rejecting candidate with {} nodes (max: {})",
                            node_count, max_nodes
                        );
                        false
                    } else {
                        true
                    }
                });
                if candidates.len() < original_count {
                    debug!(
                        "Filtered out {} candidates exceeding max node count ({})",
                        original_count - candidates.len(),
                        max_nodes
                    );
                }
            }

            if candidates.is_empty() {
                info!(
                    "No more candidates at step {} - optimization complete",
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

            // フィルタリング後の候補数を記録
            let num_candidates = candidates.len();
            trace!("Found {} candidates at step {}", num_candidates, step);

            // 候補をコスト付きで準備してSelectorで選択
            let candidates_with_cost: Vec<(AstNode, f32)> = candidates
                .into_iter()
                .map(|ast| {
                    let cost = self.estimator.estimate(&ast);
                    (ast, cost)
                })
                .collect();

            // Selectorで上位beam_width個を選択
            let selected = self.selector.select(candidates_with_cost, self.beam_width);

            beam = selected.into_iter().map(|(ast, _cost)| ast).collect();

            // このステップの最良候補を記録
            if let Some(best) = beam.first() {
                let cost = self.estimator.estimate(best);

                // コストが改善されない場合はカウンターを増やす
                if cost >= best_cost {
                    no_improvement_count += 1;
                    debug!(
                        "Step {}: no improvement (current={:.2e}, best={:.2e}, {}/{})",
                        step, cost, best_cost, no_improvement_count, MAX_NO_IMPROVEMENT_STEPS
                    );

                    // 連続で改善がない場合は早期終了
                    if no_improvement_count >= MAX_NO_IMPROVEMENT_STEPS {
                        info!(
                            "No cost improvement for {} steps - optimization complete",
                            MAX_NO_IMPROVEMENT_STEPS
                        );
                        actual_steps = step;
                        if let Some(ref pb) = pb {
                            pb.set_position(step as u64);
                            pb.set_message(format!(
                                "no cost improvement for {} steps",
                                MAX_NO_IMPROVEMENT_STEPS
                            ));
                        }
                        break;
                    }
                } else {
                    // コストが改善された場合はカウンターをリセット
                    no_improvement_count = 0;
                    let improvement_pct = (best_cost - cost) / best_cost * 100.0;
                    info!(
                        "Step {}: cost improved {:.2e} -> {:.2e} ({:+.1}%)",
                        step, best_cost, cost, -improvement_pct
                    );
                    best_cost = cost;
                    global_best = best.clone();
                }

                let step_logs = if self.collect_logs {
                    log_capture::get_captured_logs()
                } else {
                    Vec::new()
                };
                history.add_snapshot(OptimizationSnapshot::with_candidates(
                    step + 1,
                    best.clone(),
                    cost,
                    format!(
                        "Step {}: {} candidates, beam width {}",
                        step + 1,
                        num_candidates,
                        beam.len()
                    ),
                    0,
                    None,
                    step_logs,
                    num_candidates,
                ));
                // このステップのログをクリア（次のステップで新しいログのみを記録するため）
                if self.collect_logs {
                    log_capture::clear_logs();
                }
            }
        }

        if let Some(pb) = pb {
            pb.finish_and_clear();
            // Cargoスタイルの完了メッセージ
            let elapsed = start_time.elapsed();
            let time_str = if elapsed.as_secs() > 0 {
                format!("{:.2}s", elapsed.as_secs_f64())
            } else {
                format!("{}ms", elapsed.as_millis())
            };
            if actual_steps + 1 < self.max_steps {
                // 早期終了
                println!(
                    "{:>12} AST optimization in {} (converged after {} steps)",
                    "\x1b[1;32mFinished\x1b[0m",
                    time_str,
                    actual_steps + 1
                );
            } else {
                // 最大ステップ数に到達
                println!(
                    "{:>12} AST optimization in {} ({} steps)",
                    "\x1b[1;32mFinished\x1b[0m",
                    time_str,
                    actual_steps + 1
                );
            }
        }

        let final_cost = self.estimator.estimate(&global_best);
        let improvement_pct = if initial_cost > 0.0 {
            (initial_cost - final_cost) / initial_cost * 100.0
        } else {
            0.0
        };
        info!(
            "AST optimization complete: {} steps, cost {:.2e} -> {:.2e} ({:+.1}%)",
            actual_steps + 1,
            initial_cost,
            final_cost,
            -improvement_pct
        );

        // これまでで最良の候補を返す
        (global_best, history)
    }
}

impl<S, E, Sel> Optimizer for BeamSearchOptimizer<S, E, Sel>
where
    S: Suggester,
    E: CostEstimator,
    Sel: Selector<AstNode>,
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
    use crate::opt::ast::suggesters::RuleBaseSuggester;

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
    fn test_beam_search_with_max_steps_zero() {
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
