//! 枝刈り付き幅優先探索最適化器

use std::cmp::Ordering;

use super::BeamEntry;
use crate::ast::AstNode;
use crate::opt::ast::estimator::SimpleCostEstimator;
use crate::opt::ast::history::{OptimizationHistory, OptimizationSnapshot};
use crate::opt::ast::{AstCostEstimator, AstOptimizer, AstSuggester};
use crate::opt::progress::{
    FinishInfo, IndicatifProgress, NoOpProgress, ProgressState, SearchProgress,
};
use log::{info, trace};
use std::collections::VecDeque;
use std::time::Instant;

/// 枝刈り付き幅優先探索最適化器
///
/// 各レベルでコストが低い上位n個の候補のみを保持し、
/// それ以外の候補を探索しない幅優先探索です。
///
/// # 終了条件
///
/// - 最大ステップ数(`max_steps`)に達した
/// - Suggesterから新しい提案がなくなった
/// - コスト改善がない状態が続いた（`max_no_improvement_steps`）
pub struct PrunedBfsOptimizer<S, E = SimpleCostEstimator, P = IndicatifProgress>
where
    S: AstSuggester,
    E: AstCostEstimator,
    P: SearchProgress,
{
    suggester: S,
    estimator: E,
    /// 各レベルで保持する候補の最大数（枝刈り幅）
    prune_width: usize,
    max_steps: usize,
    progress: Option<P>,
    collect_logs: bool,
    max_node_count: Option<usize>,
    max_no_improvement_steps: Option<usize>,
}

impl<S> PrunedBfsOptimizer<S, SimpleCostEstimator, IndicatifProgress>
where
    S: AstSuggester,
{
    /// 新しい枝刈り付きBFS最適化器を作成
    pub fn new(suggester: S) -> Self {
        Self {
            suggester,
            estimator: SimpleCostEstimator::new(),
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

impl<S, E, P> PrunedBfsOptimizer<S, E, P>
where
    S: AstSuggester,
    E: AstCostEstimator,
    P: SearchProgress,
{
    /// カスタムコスト推定器を設定
    pub fn with_estimator<NewE>(self, estimator: NewE) -> PrunedBfsOptimizer<S, NewE, P>
    where
        NewE: AstCostEstimator,
    {
        PrunedBfsOptimizer {
            suggester: self.suggester,
            estimator,
            prune_width: self.prune_width,
            max_steps: self.max_steps,
            progress: self.progress,
            collect_logs: self.collect_logs,
            max_node_count: self.max_node_count,
            max_no_improvement_steps: self.max_no_improvement_steps,
        }
    }

    /// 枝刈り幅を設定（各レベルで保持する候補の最大数）
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
    pub fn with_progress<P2: SearchProgress>(self, progress: P2) -> PrunedBfsOptimizer<S, E, P2> {
        PrunedBfsOptimizer {
            suggester: self.suggester,
            estimator: self.estimator,
            prune_width: self.prune_width,
            max_steps: self.max_steps,
            progress: Some(progress),
            collect_logs: self.collect_logs,
            max_node_count: self.max_node_count,
            max_no_improvement_steps: self.max_no_improvement_steps,
        }
    }

    /// プログレス表示を無効化
    pub fn without_progress(self) -> PrunedBfsOptimizer<S, E, NoOpProgress> {
        PrunedBfsOptimizer {
            suggester: self.suggester,
            estimator: self.estimator,
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

    /// 候補をコストでソートして上位N件を返す
    fn select_top_n(&self, candidates: Vec<AstNode>, n: usize) -> Vec<(AstNode, f32, usize)> {
        let mut with_cost_and_index: Vec<(AstNode, f32, usize)> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, c)| {
                let cost = self.estimator.estimate(&c);
                (c, cost, idx)
            })
            .collect();
        with_cost_and_index.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        with_cost_and_index.into_iter().take(n).collect()
    }
}

impl<S, E, P> PrunedBfsOptimizer<S, E, P>
where
    S: AstSuggester,
    E: AstCostEstimator,
    P: SearchProgress,
{
    /// 履歴を記録しながら最適化を実行
    pub fn optimize_with_history(&mut self, ast: AstNode) -> (AstNode, OptimizationHistory) {
        let start_time = Instant::now();

        info!(
            "AST pruned BFS optimization started (prune_width={}, max_steps={}, max_nodes={:?})",
            self.prune_width, self.max_steps, self.max_node_count
        );

        let mut history = OptimizationHistory::new();

        // BFSキュー: (AST, path)
        let mut queue: VecDeque<BeamEntry> = VecDeque::new();
        queue.push_back(BeamEntry {
            ast: ast.clone(),
            path: vec![],
        });

        let initial_cost = self.estimator.estimate(&ast);
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
            progress.start(self.max_steps, "AST pruned BFS optimization");
        }

        for step in 0..self.max_steps {
            actual_steps = step;
            if let Some(ref mut progress) = self.progress {
                progress.update(&ProgressState::new(
                    step,
                    self.max_steps,
                    format!("step {}", step + 1),
                ));
            }

            if queue.is_empty() {
                info!("Queue empty at step {} - optimization complete", step);
                break;
            }

            // 現在のレベルの全ノードから候補を生成
            let current_level_size = queue.len();
            let mut all_candidates: Vec<(AstNode, String, String, super::OptimizationPath)> =
                Vec::new();

            for _ in 0..current_level_size {
                if let Some(entry) = queue.pop_front() {
                    let new_candidates = self.suggester.suggest(&entry.ast);
                    for result in new_candidates {
                        let mut new_path = entry.path.clone();
                        new_path.push((result.suggester_name.clone(), result.description.clone()));
                        all_candidates.push((
                            result.ast,
                            result.suggester_name,
                            result.description,
                            new_path,
                        ));
                    }
                }
            }

            // 最大ノード数制限を適用
            if let Some(max_nodes) = self.max_node_count {
                all_candidates
                    .retain(|(ast, _, _, _)| SimpleCostEstimator::get_node_count(ast) <= max_nodes);
            }

            if all_candidates.is_empty() {
                info!(
                    "No more candidates at step {} - optimization complete",
                    step
                );
                break;
            }

            let num_candidates = all_candidates.len();
            trace!("Found {} candidates at step {}", num_candidates, step);

            // ASTのみを取り出してコスト計算・ソート
            let candidates: Vec<AstNode> = all_candidates
                .iter()
                .map(|(ast, _, _, _)| ast.clone())
                .collect();
            let sorted = self.select_top_n(candidates, num_candidates);

            // 枝刈り: 上位prune_width個のみを次のレベルへ
            for (ast, cost, idx) in sorted.iter().take(self.prune_width) {
                let (_, _, _, path) = &all_candidates[*idx];
                queue.push_back(BeamEntry {
                    ast: ast.clone(),
                    path: path.clone(),
                });

                // 最良コストを更新
                if *cost < global_best_cost {
                    no_improvement_count = 0;
                    let improvement_pct = (global_best_cost - cost) / global_best_cost * 100.0;
                    info!(
                        "Step {}: cost improved {:.2e} -> {:.2e} ({:+.1}%)",
                        step, global_best_cost, cost, -improvement_pct
                    );
                    global_best_cost = *cost;
                    global_best = BeamEntry {
                        ast: ast.clone(),
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
                }
            }

            // コスト改善がない場合のカウンター
            if sorted
                .first()
                .map(|(_, c, _)| *c >= global_best_cost)
                .unwrap_or(true)
            {
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

        if let Some(ref mut progress) = self.progress {
            let elapsed = start_time.elapsed();
            let finish_info = FinishInfo::new(
                elapsed,
                actual_steps + 1,
                self.max_steps,
                "AST pruned BFS optimization",
            );
            progress.finish(&finish_info);
        }

        let improvement_pct = if initial_cost > 0.0 {
            (initial_cost - global_best_cost) / initial_cost * 100.0
        } else {
            0.0
        };
        info!(
            "AST pruned BFS complete: {} steps, cost {:.2e} -> {:.2e} ({:+.1}%)",
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

impl<S, E, P> AstOptimizer for PrunedBfsOptimizer<S, E, P>
where
    S: AstSuggester,
    E: AstCostEstimator,
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
    fn test_pruned_bfs_optimizer() {
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

        let mut optimizer = PrunedBfsOptimizer::new(suggester)
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
    fn test_pruned_bfs_complex() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);

        let mut optimizer = PrunedBfsOptimizer::new(suggester)
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
    fn test_pruned_bfs_no_applicable_rules() {
        // マッチしないルールのみ
        let rule = astpat!(|a| {
            AstNode::Mul(Box::new(a), Box::new(AstNode::Const(Literal::I64(99))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule]);

        let mut optimizer = PrunedBfsOptimizer::new(suggester)
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
    fn test_pruned_bfs_with_prune_width_one() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);

        // 枝刈り幅1（貪欲法に近い）
        let mut optimizer = PrunedBfsOptimizer::new(suggester)
            .with_prune_width(1)
            .with_max_steps(10)
            .without_progress();

        let input = const_int(5) + const_int(0);

        let result = optimizer.optimize(input);
        // 枝刈り幅1でも最適化できるはず
        assert_eq!(result, const_int(5));
    }
}
