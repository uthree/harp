use super::selector::{AstCostSelector, AstSelector};
use crate::ast::AstNode;
use crate::ast::pat::{AstRewriteRule, AstRewriter};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, trace};
use std::rc::Rc;
use std::time::Instant;

use super::history::{AlternativeCandidate, OptimizationHistory, OptimizationSnapshot};
use super::{AstCostEstimator, AstOptimizer, AstSuggester};

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

impl AstOptimizer for RuleBaseOptimizer {
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
/// `AstSelector`により候補選択・コスト評価を行います。
/// デフォルトは`AstCostSelector`（SimpleCostEstimatorでコスト計算、上位n件を選択）。
/// `RuntimeSelector`を使用すると、静的コストで足切り後に実行時間を計測して選択します。
pub struct BeamSearchOptimizer<S, Sel = AstCostSelector>
where
    S: AstSuggester,
    Sel: AstSelector,
{
    suggester: S,
    selector: Sel,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
    collect_logs: bool,
    max_node_count: Option<usize>,
    /// 改善がない場合に早期終了するまでのステップ数（Noneで無効化）
    max_no_improvement_steps: Option<usize>,
}

impl<S> BeamSearchOptimizer<S, AstCostSelector>
where
    S: AstSuggester,
{
    /// 新しいビームサーチ最適化器を作成
    ///
    /// デフォルトでは`AstCostSelector`を使用します。
    /// コスト評価はSelector自身が行います。
    pub fn new(suggester: S) -> Self {
        Self {
            suggester,
            selector: AstCostSelector::new(),
            beam_width: 10,
            max_steps: 10000,
            show_progress: cfg!(debug_assertions),
            collect_logs: cfg!(debug_assertions),
            max_node_count: Some(10000), // デフォルトで最大1万ノードに制限
            max_no_improvement_steps: Some(3), // デフォルトで3ステップ改善なしで終了
        }
    }
}

impl<S, Sel> BeamSearchOptimizer<S, Sel>
where
    S: AstSuggester,
    Sel: AstSelector,
{
    /// カスタム選択器を設定
    ///
    /// デフォルトの`AstCostSelector`の代わりに、
    /// `RuntimeSelector`などのカスタム選択器を使用できます。
    ///
    /// # Example
    ///
    /// ```ignore
    /// use harp::opt::{RuntimeSelector, BeamSearchOptimizer};
    ///
    /// let selector = RuntimeSelector::new(renderer, compiler, signature, buffer_factory)
    ///     .with_pre_filter_count(10);
    ///
    /// let optimizer = BeamSearchOptimizer::new(suggester)
    ///     .with_selector(selector);
    /// ```
    pub fn with_selector<NewSel>(self, selector: NewSel) -> BeamSearchOptimizer<S, NewSel>
    where
        NewSel: AstSelector,
    {
        BeamSearchOptimizer {
            suggester: self.suggester,
            selector,
            beam_width: self.beam_width,
            max_steps: self.max_steps,
            show_progress: self.show_progress,
            collect_logs: self.collect_logs,
            max_node_count: self.max_node_count,
            max_no_improvement_steps: self.max_no_improvement_steps,
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

    /// コスト改善がない場合の早期終了ステップ数を設定
    ///
    /// 連続して指定したステップ数だけコストが改善されない場合、最適化を早期終了します。
    /// Noneを指定すると早期終了を無効化します（max_stepsまで実行）。
    ///
    /// デフォルト: Some(3)
    pub fn with_no_improvement_limit(mut self, steps: Option<usize>) -> Self {
        self.max_no_improvement_steps = steps;
        self
    }
}

/// 最適化パス：(suggester_name, description)のリスト
type OptimizationPath = Vec<(String, String)>;

/// 候補情報: (suggester_name, description, path)
type CandidateInfo = (String, String, OptimizationPath);

/// ビームエントリ：ASTと、そのASTに至るまでの変換パス
#[derive(Clone, Debug)]
struct BeamEntry {
    ast: AstNode,
    /// このASTに至るまでの変換パス（各ステップでの(suggester_name, description)）
    path: OptimizationPath,
}

impl<S, Sel> BeamSearchOptimizer<S, Sel>
where
    S: AstSuggester,
    Sel: AstSelector,
{
    /// 履歴を記録しながら最適化を実行
    ///
    /// コスト評価はSelectorが行います。履歴記録用のコストはSelectorから返される値を使用します。
    pub fn optimize_with_history(&self, ast: AstNode) -> (AstNode, OptimizationHistory) {
        use crate::opt::ast::estimator::SimpleCostEstimator;
        use crate::opt::log_capture;

        // 履歴記録用の静的コスト推定器
        let static_estimator = SimpleCostEstimator::new();

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
        // ビームエントリに変換パスを追加
        let mut beam = vec![BeamEntry {
            ast: ast.clone(),
            path: vec![],
        }];

        // 初期状態を記録（静的コストを使用）
        let initial_cost = static_estimator.estimate(&ast);
        info!("Initial AST cost: {:.2e}", initial_cost);
        let initial_logs = if self.collect_logs {
            log_capture::get_captured_logs()
        } else {
            Vec::new()
        };

        // これまでで最良の候補を保持（astを移動する前に初期化）
        let mut global_best = BeamEntry {
            ast: ast.clone(),
            path: vec![],
        };
        let mut global_best_cost = initial_cost;

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

        for step in 0..self.max_steps {
            actual_steps = step;
            if let Some(ref pb) = pb {
                pb.set_message(format!("step {}", step + 1));
                pb.set_position(step as u64);
            }

            // 候補: (AST, suggester_name, description, parent_path)
            let mut candidates_with_info: Vec<(AstNode, String, String, OptimizationPath)> =
                Vec::new();

            // 現在のビーム内の各候補から新しい候補を生成
            for entry in &beam {
                let new_candidates = self.suggester.suggest(&entry.ast);
                for result in new_candidates {
                    // 親のパスを継承し、現在の変換を追加
                    let mut new_path = entry.path.clone();
                    new_path.push((result.suggester_name.clone(), result.description.clone()));
                    candidates_with_info.push((
                        result.ast,
                        result.suggester_name,
                        result.description,
                        new_path,
                    ));
                }
            }

            // 最大ノード数制限を適用
            let original_count = candidates_with_info.len();
            if let Some(max_nodes) = self.max_node_count {
                candidates_with_info.retain(|(ast, _, _, _)| {
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
                if candidates_with_info.len() < original_count {
                    debug!(
                        "Filtered out {} candidates exceeding max node count ({})",
                        original_count - candidates_with_info.len(),
                        max_nodes
                    );
                }
            }

            if candidates_with_info.is_empty() {
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
            let num_candidates = candidates_with_info.len();
            trace!("Found {} candidates at step {}", num_candidates, step);

            // 候補情報を分離（インデックスベースで管理）
            let candidate_infos: Vec<CandidateInfo> = candidates_with_info
                .iter()
                .map(|(_, name, desc, path)| (name.clone(), desc.clone(), path.clone()))
                .collect();

            // 候補のASTだけを取り出してSelectorに渡す
            let candidates: Vec<AstNode> = candidates_with_info
                .into_iter()
                .map(|(ast, _, _, _)| ast)
                .collect();

            // Selectorで全候補のコストを計算してソート（上位全件取得）
            // select_with_indicesを使用してインデックスを保持
            let all_with_cost_and_index = self
                .selector
                .select_with_indices(candidates, num_candidates);

            // ビーム用に上位beam_width個を取得
            let selected: Vec<_> = all_with_cost_and_index
                .iter()
                .take(self.beam_width)
                .cloned()
                .collect();

            // 新しいビームを構築（パス情報を保持）
            beam = selected
                .iter()
                .map(|(ast, _, idx)| {
                    let path = candidate_infos
                        .get(*idx)
                        .map(|(_, _, p)| p.clone())
                        .unwrap_or_default();
                    BeamEntry {
                        ast: ast.clone(),
                        path,
                    }
                })
                .collect();

            // このステップの最良候補を処理
            if let Some((best, cost, best_index)) = selected.first() {
                // 最良候補のパス情報を取得
                let best_path = candidate_infos
                    .get(*best_index)
                    .map(|(_, _, p)| p.clone())
                    .unwrap_or_default();

                // コストが改善されない場合はカウンターを増やす
                if *cost >= best_cost {
                    no_improvement_count += 1;

                    // 早期終了の判定
                    if let Some(max_no_improvement) = self.max_no_improvement_steps {
                        debug!(
                            "Step {}: no improvement (current={:.2e}, best={:.2e}, {}/{})",
                            step, cost, best_cost, no_improvement_count, max_no_improvement
                        );

                        // 連続で改善がない場合は早期終了
                        if no_improvement_count >= max_no_improvement {
                            info!(
                                "No cost improvement for {} steps - optimization complete",
                                max_no_improvement
                            );
                            actual_steps = step;
                            if let Some(ref pb) = pb {
                                pb.set_position(step as u64);
                                pb.set_message(format!(
                                    "no cost improvement for {} steps",
                                    max_no_improvement
                                ));
                            }
                            break;
                        }
                    } else {
                        debug!(
                            "Step {}: no improvement (current={:.2e}, best={:.2e})",
                            step, cost, best_cost
                        );
                    }

                    // ログをクリア（次のステップで新しいログのみを記録するため）
                    if self.collect_logs {
                        log_capture::clear_logs();
                    }
                } else {
                    // コストが改善された場合のみスナップショットを記録
                    no_improvement_count = 0;
                    let improvement_pct = (best_cost - cost) / best_cost * 100.0;
                    info!(
                        "Step {}: cost improved {:.2e} -> {:.2e} ({:+.1}%)",
                        step, best_cost, cost, -improvement_pct
                    );
                    best_cost = *cost;
                    global_best = BeamEntry {
                        ast: best.clone(),
                        path: best_path.clone(),
                    };
                    global_best_cost = *cost;

                    // ログを取得
                    let step_logs = if self.collect_logs {
                        log_capture::get_captured_logs()
                    } else {
                        Vec::new()
                    };

                    // global_bestのパスからSuggester名を取得（最後に適用されたSuggester）
                    let suggester_name = global_best.path.last().map(|(name, _)| name.clone());

                    // 代替候補を構築（rank > 0の全候補、ビームに入らなかったものも含む）
                    let alternatives: Vec<AlternativeCandidate> = all_with_cost_and_index
                        .iter()
                        .skip(1)
                        .enumerate()
                        .map(|(idx, (ast, cost, original_index))| {
                            let (_, desc, path) = candidate_infos
                                .get(*original_index)
                                .cloned()
                                .unwrap_or_default();
                            // パスの最後の要素からSuggester名を取得
                            let name = path.last().map(|(n, _)| n.clone()).unwrap_or_default();
                            AlternativeCandidate {
                                ast: ast.clone(),
                                cost: *cost,
                                suggester_name: Some(name),
                                description: desc,
                                rank: idx + 1,
                            }
                        })
                        .collect();

                    // スナップショットのステップ番号は履歴の現在の長さ（連続した番号）
                    let snapshot_step = history.len();

                    history.add_snapshot(OptimizationSnapshot::with_alternatives(
                        snapshot_step,
                        global_best.ast.clone(),
                        global_best_cost,
                        format!(
                            "Step {}: {} candidates, cost improved",
                            snapshot_step, num_candidates
                        ),
                        0,
                        None,
                        step_logs,
                        num_candidates,
                        suggester_name,
                        alternatives,
                        global_best.path.clone(),
                    ));

                    // ログをクリア（次のステップで新しいログのみを記録するため）
                    if self.collect_logs {
                        log_capture::clear_logs();
                    }
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

        let improvement_pct = if initial_cost > 0.0 {
            (initial_cost - global_best_cost) / initial_cost * 100.0
        } else {
            0.0
        };
        info!(
            "AST optimization complete: {} steps, cost {:.2e} -> {:.2e} ({:+.1}%)",
            actual_steps + 1,
            initial_cost,
            global_best_cost,
            -improvement_pct
        );

        // 最終結果が履歴の最後と異なる場合（コスト改善がなかったステップが最後の場合）、
        // global_bestを最終スナップショットとして追加
        let last_snapshot_cost = history
            .snapshots()
            .last()
            .map(|s| s.cost)
            .unwrap_or(f32::MAX);
        if (global_best_cost - last_snapshot_cost).abs() > f32::EPSILON {
            let final_logs = if self.collect_logs {
                log_capture::get_captured_logs()
            } else {
                Vec::new()
            };
            let final_step = history.len();
            history.add_snapshot(OptimizationSnapshot::with_candidates(
                final_step,
                global_best.ast.clone(),
                global_best_cost,
                format!(
                    "[Final] Best result (cost={:.2e}, improved {:.1}%)",
                    global_best_cost, improvement_pct
                ),
                0,
                None,
                final_logs,
                0,
            ));
        }

        // 最終結果のパス情報を履歴に記録
        history.set_final_path(global_best.path.clone());

        // 最終結果のパス情報をログに出力（デバッグ用）
        if !global_best.path.is_empty() {
            debug!(
                "Final optimization path ({} steps): {}",
                global_best.path.len(),
                global_best
                    .path
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            );
        }

        // これまでで最良の候補を返す
        (global_best.ast, history)
    }
}

impl<S, Sel> AstOptimizer for BeamSearchOptimizer<S, Sel>
where
    S: AstSuggester,
    Sel: AstSelector,
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

        let optimizer = BeamSearchOptimizer::new(suggester)
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

        let optimizer = BeamSearchOptimizer::new(suggester)
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

        let optimizer = BeamSearchOptimizer::new(suggester)
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

        let optimizer = BeamSearchOptimizer::new(suggester)
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

        // ビーム幅1（貪欲法）
        let optimizer = BeamSearchOptimizer::new(suggester)
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

        // 最大ステップ数0（最適化しない）
        let optimizer = BeamSearchOptimizer::new(suggester)
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

        let optimizer = BeamSearchOptimizer::new(suggester)
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

        // 非常に大きなビーム幅
        let optimizer = BeamSearchOptimizer::new(suggester)
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
