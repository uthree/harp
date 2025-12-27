//! 枝刈り付き深さ優先探索最適化器

use super::BeamEntry;
use crate::ast::AstNode;
use crate::opt::ast::history::{OptimizationHistory, OptimizationSnapshot};
use crate::opt::ast::selector::{AstCostSelector, AstSelector};
use crate::opt::ast::{AstCostEstimator, AstOptimizer, AstSuggester};
use crate::opt::progress::{
    FinishInfo, IndicatifProgress, NoOpProgress, ProgressState, SearchProgress,
};
use log::{info, trace};
use std::time::Instant;

/// 枝刈り付き深さ優先探索最適化器
///
/// 各ノードでコストが低い上位n個の子候補のみを探索し、
/// それ以外の候補を探索しない深さ優先探索です。
///
/// 有望な候補を先に深く探索するため、
/// 早期に良い解を見つけやすい特性があります。
///
/// # 終了条件
///
/// - 最大ステップ数(`max_steps`)に達した
/// - Suggesterから新しい提案がなくなった
/// - コスト改善がない状態が続いた（`max_no_improvement_steps`）
pub struct PrunedDfsOptimizer<S, Sel = AstCostSelector, P = IndicatifProgress>
where
    S: AstSuggester,
    Sel: AstSelector,
    P: SearchProgress,
{
    suggester: S,
    selector: Sel,
    /// 各ノードで探索する子候補の最大数（枝刈り幅）
    prune_width: usize,
    max_steps: usize,
    progress: Option<P>,
    collect_logs: bool,
    max_node_count: Option<usize>,
    max_no_improvement_steps: Option<usize>,
}

impl<S> PrunedDfsOptimizer<S, AstCostSelector, IndicatifProgress>
where
    S: AstSuggester,
{
    /// 新しい枝刈り付きDFS最適化器を作成
    pub fn new(suggester: S) -> Self {
        Self {
            suggester,
            selector: AstCostSelector::new(),
            prune_width: 10,
            max_steps: 10000,
            progress: if cfg!(debug_assertions) {
                Some(IndicatifProgress::new())
            } else {
                None
            },
            collect_logs: cfg!(debug_assertions),
            max_node_count: Some(10000),
            max_no_improvement_steps: Some(3),
        }
    }
}

impl<S, Sel, P> PrunedDfsOptimizer<S, Sel, P>
where
    S: AstSuggester,
    Sel: AstSelector,
    P: SearchProgress,
{
    /// カスタム選択器を設定
    pub fn with_selector<NewSel>(self, selector: NewSel) -> PrunedDfsOptimizer<S, NewSel, P>
    where
        NewSel: AstSelector,
    {
        PrunedDfsOptimizer {
            suggester: self.suggester,
            selector,
            prune_width: self.prune_width,
            max_steps: self.max_steps,
            progress: self.progress,
            collect_logs: self.collect_logs,
            max_node_count: self.max_node_count,
            max_no_improvement_steps: self.max_no_improvement_steps,
        }
    }

    /// 枝刈り幅を設定（各ノードで探索する子候補の最大数）
    pub fn with_prune_width(mut self, width: usize) -> Self {
        self.prune_width = width;
        self
    }

    /// 最大ステップ数を設定
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// カスタムプログレス表示器を設定
    pub fn with_progress<P2: SearchProgress>(self, progress: P2) -> PrunedDfsOptimizer<S, Sel, P2> {
        PrunedDfsOptimizer {
            suggester: self.suggester,
            selector: self.selector,
            prune_width: self.prune_width,
            max_steps: self.max_steps,
            progress: Some(progress),
            collect_logs: self.collect_logs,
            max_node_count: self.max_node_count,
            max_no_improvement_steps: self.max_no_improvement_steps,
        }
    }

    /// プログレス表示を無効化
    pub fn without_progress(self) -> PrunedDfsOptimizer<S, Sel, NoOpProgress> {
        PrunedDfsOptimizer {
            suggester: self.suggester,
            selector: self.selector,
            prune_width: self.prune_width,
            max_steps: self.max_steps,
            progress: None,
            collect_logs: self.collect_logs,
            max_node_count: self.max_node_count,
            max_no_improvement_steps: self.max_no_improvement_steps,
        }
    }

    /// 最大ノード数を設定
    pub fn with_max_node_count(mut self, max: Option<usize>) -> Self {
        self.max_node_count = max;
        self
    }

    /// コスト改善がない場合の早期終了ステップ数を設定
    pub fn with_no_improvement_limit(mut self, steps: Option<usize>) -> Self {
        self.max_no_improvement_steps = steps;
        self
    }

    /// ログ収集の有効/無効を設定
    pub fn with_collect_logs(mut self, collect: bool) -> Self {
        self.collect_logs = collect;
        self
    }
}

impl<S, Sel, P> PrunedDfsOptimizer<S, Sel, P>
where
    S: AstSuggester,
    Sel: AstSelector,
    P: SearchProgress,
{
    /// 履歴を記録しながら最適化を実行
    pub fn optimize_with_history(&mut self, ast: AstNode) -> (AstNode, OptimizationHistory) {
        use crate::opt::ast::estimator::SimpleCostEstimator;

        let start_time = Instant::now();
        let static_estimator = SimpleCostEstimator::new();

        info!(
            "AST pruned DFS optimization started (prune_width={}, max_steps={}, max_nodes={:?})",
            self.prune_width, self.max_steps, self.max_node_count
        );

        let mut history = OptimizationHistory::new();

        // DFSスタック: (AST, path)
        let mut stack: Vec<BeamEntry> = vec![BeamEntry {
            ast: ast.clone(),
            path: vec![],
        }];

        let initial_cost = static_estimator.estimate(&ast);
        info!("Initial AST cost: {:.2e}", initial_cost);

        // 初期状態を記録
        history.add_snapshot(OptimizationSnapshot::with_candidates(
            0,
            ast.clone(),
            initial_cost,
            "Initial AST".to_string(),
            0,
            None,
            Vec::new(),
            0,
        ));

        let mut global_best = BeamEntry {
            ast: ast.clone(),
            path: vec![],
        };
        let mut global_best_cost = initial_cost;
        let mut no_improvement_count = 0;
        let mut actual_steps = 0;

        if let Some(ref mut progress) = self.progress {
            progress.start(self.max_steps, "AST pruned DFS optimization");
        }

        for step in 0..self.max_steps {
            actual_steps = step;
            if let Some(ref mut progress) = self.progress {
                progress.update(&ProgressState::new(
                    step,
                    self.max_steps,
                    format!("step {} (stack={})", step + 1, stack.len()),
                ));
            }

            // スタックから1つ取り出す
            let Some(entry) = stack.pop() else {
                info!("Stack empty at step {} - optimization complete", step);
                break;
            };

            // 候補を生成
            let new_candidates: Vec<_> = self
                .suggester
                .suggest(&entry.ast)
                .into_iter()
                .map(|result| {
                    let mut new_path = entry.path.clone();
                    new_path.push((result.suggester_name.clone(), result.description.clone()));
                    (
                        result.ast,
                        result.suggester_name,
                        result.description,
                        new_path,
                    )
                })
                .collect();

            // 最大ノード数制限を適用
            let filtered_candidates: Vec<_> = if let Some(max_nodes) = self.max_node_count {
                new_candidates
                    .into_iter()
                    .filter(|(ast, _, _, _)| SimpleCostEstimator::get_node_count(ast) <= max_nodes)
                    .collect()
            } else {
                new_candidates
            };

            if filtered_candidates.is_empty() {
                // このノードからは候補がない、バックトラック
                trace!("No candidates from current node, backtracking");
                continue;
            }

            let num_candidates = filtered_candidates.len();
            trace!("Found {} candidates at step {}", num_candidates, step);

            // ASTのみを取り出してSelectorでソート
            let candidates: Vec<AstNode> = filtered_candidates
                .iter()
                .map(|(ast, _, _, _)| ast.clone())
                .collect();
            let sorted = self
                .selector
                .select_with_indices(candidates, num_candidates);

            // 最良の候補でグローバルベストを更新
            if let Some((best_ast, best_cost, best_idx)) = sorted.first() {
                let (_, _, _, path) = &filtered_candidates[*best_idx];
                if *best_cost < global_best_cost {
                    no_improvement_count = 0;
                    let improvement_pct = (global_best_cost - best_cost) / global_best_cost * 100.0;
                    info!(
                        "Step {}: cost improved {:.2e} -> {:.2e} ({:+.1}%)",
                        step, global_best_cost, best_cost, -improvement_pct
                    );
                    global_best_cost = *best_cost;
                    global_best = BeamEntry {
                        ast: best_ast.clone(),
                        path: path.clone(),
                    };

                    // コスト改善時にスナップショットを記録
                    let suggester_name = path.last().map(|(name, _)| name.clone());
                    let snapshot_step = history.len();
                    history.add_snapshot(OptimizationSnapshot::with_candidates(
                        snapshot_step,
                        global_best.ast.clone(),
                        global_best_cost,
                        format!(
                            "Step {}: {} candidates, cost improved",
                            snapshot_step, num_candidates
                        ),
                        0,
                        suggester_name,
                        Vec::new(),
                        num_candidates,
                    ));
                } else {
                    no_improvement_count += 1;
                    if let Some(max_no_improvement) = self.max_no_improvement_steps
                        && no_improvement_count >= max_no_improvement
                    {
                        info!(
                            "No cost improvement for {} steps - optimization complete",
                            max_no_improvement
                        );
                        break;
                    }
                }
            }

            // 枝刈り: 上位prune_width個をスタックに追加（逆順で追加して最良を先に処理）
            let pruned: Vec<_> = sorted.iter().take(self.prune_width).collect();
            for (ast, _, idx) in pruned.into_iter().rev() {
                let (_, _, _, path) = &filtered_candidates[*idx];
                stack.push(BeamEntry {
                    ast: ast.clone(),
                    path: path.clone(),
                });
            }
        }

        if let Some(ref mut progress) = self.progress {
            let elapsed = start_time.elapsed();
            let finish_info = FinishInfo::new(
                elapsed,
                actual_steps + 1,
                self.max_steps,
                "AST pruned DFS optimization",
            );
            progress.finish(&finish_info);
        }

        let improvement_pct = if initial_cost > 0.0 {
            (initial_cost - global_best_cost) / initial_cost * 100.0
        } else {
            0.0
        };
        info!(
            "AST pruned DFS complete: {} steps, cost {:.2e} -> {:.2e} ({:+.1}%)",
            actual_steps + 1,
            initial_cost,
            global_best_cost,
            -improvement_pct
        );

        // 最終結果のパス情報を履歴に記録
        history.set_final_path(global_best.path.clone());

        (global_best.ast, history)
    }
}

impl<S, Sel, P> AstOptimizer for PrunedDfsOptimizer<S, Sel, P>
where
    S: AstSuggester,
    Sel: AstSelector,
    P: SearchProgress,
{
    fn optimize(&mut self, ast: AstNode) -> AstNode {
        let (optimized, _) = self.optimize_with_history(ast);
        optimized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;
    use crate::ast::helper::const_int;
    use crate::astpat;
    use crate::opt::ast::suggesters::RuleBaseSuggester;

    #[test]
    fn test_pruned_dfs_optimizer() {
        // 交換則と単位元除去のルール
        let rule1 = astpat!(|a, b| {
            AstNode::Add(Box::new(a), Box::new(b))
        } => {
            AstNode::Add(Box::new(b), Box::new(a))
        });

        let rule2 = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::I64(0))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule1, rule2]);

        let mut optimizer = PrunedDfsOptimizer::new(suggester)
            .with_prune_width(5)
            .with_max_steps(5)
            .without_progress();

        // (42 + 0) を最適化
        let input = const_int(42) + const_int(0);

        let result = optimizer.optimize(input);
        // 最終的に42に簡約されるはず
        assert_eq!(result, const_int(42));
    }

    #[test]
    fn test_pruned_dfs_complex() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);

        let mut optimizer = PrunedDfsOptimizer::new(suggester)
            .with_prune_width(10)
            .with_max_steps(10)
            .without_progress();

        // ((2 + 3) * 1) + 0 を最適化
        let input = ((const_int(2) + const_int(3)) * const_int(1)) + const_int(0);

        let result = optimizer.optimize(input);
        // 最終的に5に簡約されるはず
        assert_eq!(result, const_int(5));
    }

    #[test]
    fn test_pruned_dfs_no_applicable_rules() {
        // マッチしないルールのみ
        let rule = astpat!(|a| {
            AstNode::Mul(Box::new(a), Box::new(AstNode::Const(Literal::I64(99))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule]);

        let mut optimizer = PrunedDfsOptimizer::new(suggester)
            .with_prune_width(5)
            .with_max_steps(5)
            .without_progress();

        // ルールが適用されない入力
        let input = const_int(42);
        let result = optimizer.optimize(input.clone());

        // 変更されないはず
        assert_eq!(result, input);
    }

    #[test]
    fn test_pruned_dfs_with_prune_width_one() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);

        // 枝刈り幅1（深さ優先で最良候補のみ探索）
        let mut optimizer = PrunedDfsOptimizer::new(suggester)
            .with_prune_width(1)
            .with_max_steps(10)
            .without_progress();

        let input = const_int(5) + const_int(0);

        let result = optimizer.optimize(input);
        // 枝刈り幅1でも最適化できるはず
        assert_eq!(result, const_int(5));
    }

    #[test]
    fn test_pruned_dfs_early_termination() {
        // 1回で最適化が完了し、それ以降候補がなくなるケース
        let rule = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::I64(0))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule]);

        let mut optimizer = PrunedDfsOptimizer::new(suggester)
            .with_prune_width(5)
            .with_max_steps(10) // 最大ステップ数は10だが早期終了するはず
            .without_progress();

        let input = const_int(42) + const_int(0);

        let result = optimizer.optimize(input);
        assert_eq!(result, const_int(42));
    }
}
