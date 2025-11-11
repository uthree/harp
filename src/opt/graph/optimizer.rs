use crate::graph::Graph;
use crate::opt::graph::{
    GraphCostEstimator, GraphOptimizer, GraphSuggester, OptimizationHistory, OptimizationSnapshot,
};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;

/// ビームサーチグラフ最適化器
///
/// # 終了条件
///
/// 最適化は以下のいずれかの条件で終了します：
/// - 最大ステップ数(`max_steps`)に達した
/// - Suggesterから新しい提案がなくなった（これ以上最適化できない）
pub struct BeamSearchGraphOptimizer<S, E>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
{
    suggester: S,
    estimator: E,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
}

impl<S, E> BeamSearchGraphOptimizer<S, E>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
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

impl<S, E> BeamSearchGraphOptimizer<S, E>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
{
    /// グラフを最適化して、グラフと最適化履歴を返す
    pub fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        use crate::opt::log_capture;

        // ログキャプチャを開始
        log_capture::start_capture();

        debug!("BeamSearchGraphOptimizer: Starting beam search optimization with history tracking");

        let mut history = OptimizationHistory::new();
        let mut beam = vec![graph.clone()];

        // 初期状態を記録
        let initial_cost = self.estimator.estimate(&graph);
        let initial_outputs = graph.outputs().len();

        // 千日手対策用の変数
        let mut best_cost = initial_cost;
        let mut no_improvement_count = 0;
        const MAX_NO_IMPROVEMENT: usize = 10; // 10回連続で改善がなければ終了

        // 初期状態の入力・出力ノード情報をログに出力
        debug!(
            "BeamSearchGraphOptimizer: Initial - {} inputs, {} outputs",
            graph.inputs().len(),
            graph.outputs().len()
        );
        for (name, node) in graph.outputs() {
            let op_type = format!("{:?}", node.op);
            debug!(
                "BeamSearchGraphOptimizer: Initial - Output '{}': {:?}",
                name, op_type
            );
        }

        let initial_logs = log_capture::get_captured_logs();
        history.add_snapshot(OptimizationSnapshot::with_logs(
            0,
            graph,
            initial_cost,
            format!("Initial graph ({} outputs)", initial_outputs),
            initial_logs,
        ));

        // 初期ログをクリア（各ステップで新しいログのみを記録するため）
        log_capture::clear_logs();

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_steps as u64);

            // Cargoスタイルのプログレスバー
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

        let mut early_terminated = false;

        for step in 0..self.max_steps {
            if let Some(ref pb) = pb {
                pb.set_message(format!("step {}", step + 1));
                pb.set_position(step as u64);
            }

            let mut candidates = Vec::new();

            // 現在のビーム内の各候補から新しい候補を生成
            for graph in &beam {
                let new_candidates = self.suggester.suggest(graph);
                candidates.extend(new_candidates);
            }

            if candidates.is_empty() {
                debug!(
                    "BeamSearchGraphOptimizer: No more candidates at step {} - optimization complete (early termination)",
                    step
                );
                if let Some(ref pb) = pb {
                    pb.finish_and_clear();
                    println!(
                        "{:>12} graph optimization (no more candidates)",
                        "\x1b[1;32mFinished\x1b[0m"
                    );
                }
                early_terminated = true;
                break;
            }

            debug!(
                "BeamSearchGraphOptimizer: Found {} candidates at step {}",
                candidates.len(),
                step
            );

            // 重複除去: グラフをシリアライズして比較
            use std::collections::HashMap;

            // キャッシュのサイズ制限（メモリ消費を抑える）
            const MAX_CACHE_SIZE: usize = 10000;

            let mut seen = HashMap::new();
            let mut unique_candidates = Vec::new();

            for graph in candidates {
                // グラフの構造を文字列化（DOT形式）
                let signature = graph.to_dot();

                // キャッシュサイズが制限を超えたらクリア
                if seen.len() >= MAX_CACHE_SIZE {
                    debug!(
                        "BeamSearchGraphOptimizer: Cache size limit reached ({}), clearing cache",
                        MAX_CACHE_SIZE
                    );
                    seen.clear();
                }

                if seen.insert(signature, ()).is_none() {
                    unique_candidates.push(graph);
                }
            }

            debug!(
                "BeamSearchGraphOptimizer: After deduplication: {} unique candidates (cache size: {})",
                unique_candidates.len(),
                seen.len()
            );

            // コストでソートして上位beam_width個を残す
            // 各候補のコストを事前に計算してキャッシュ（重複計算を避ける）
            let mut candidates_with_cost: Vec<(Graph, f32)> = unique_candidates
                .into_iter()
                .map(|g| {
                    let cost = self.estimator.estimate(&g);
                    (g, cost)
                })
                .collect();

            candidates_with_cost
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // 上位beam_width個を取得（コスト情報も保持）
            let top_candidates: Vec<(Graph, f32)> = candidates_with_cost
                .into_iter()
                .take(self.beam_width)
                .collect();

            beam = top_candidates.iter().map(|(g, _)| g.clone()).collect();

            // このステップの最良候補を記録（既に計算したコストを再利用）
            if let Some((best, cost)) = top_candidates.first() {
                let num_outputs = best.outputs().len();
                let num_inputs = best.inputs().len();

                // 入力・出力ノード数をログに出力
                debug!(
                    "BeamSearchGraphOptimizer: Step {} - {} inputs, {} outputs",
                    step + 1,
                    num_inputs,
                    num_outputs
                );

                // 出力ノードの演算タイプもログに出力
                for (name, node) in best.outputs() {
                    let op_type = format!("{:?}", node.op);
                    debug!(
                        "BeamSearchGraphOptimizer: Step {} - Output '{}': {:?}",
                        step + 1,
                        name,
                        op_type
                    );
                }

                let step_logs = log_capture::get_captured_logs();
                history.add_snapshot(OptimizationSnapshot::with_logs(
                    step + 1,
                    best.clone(),
                    *cost,
                    format!(
                        "Step {}: beam width {}, outputs: {}",
                        step + 1,
                        beam.len(),
                        num_outputs
                    ),
                    step_logs,
                ));

                // このステップのログをクリア（次のステップで新しいログのみを記録するため）
                log_capture::clear_logs();

                // コスト改善チェック（千日手対策）
                const EPSILON: f32 = 1e-6; // 浮動小数点の誤差を考慮
                if *cost < best_cost - EPSILON {
                    // コストが改善された
                    best_cost = *cost;
                    no_improvement_count = 0;
                    debug!(
                        "BeamSearchGraphOptimizer: Cost improved to {} at step {}",
                        cost,
                        step + 1
                    );
                } else {
                    // コストが改善されなかった
                    no_improvement_count += 1;
                    debug!(
                        "BeamSearchGraphOptimizer: No cost improvement at step {} (count: {}/{})",
                        step + 1,
                        no_improvement_count,
                        MAX_NO_IMPROVEMENT
                    );

                    if no_improvement_count >= MAX_NO_IMPROVEMENT {
                        debug!(
                            "BeamSearchGraphOptimizer: No improvement for {} steps - early termination",
                            MAX_NO_IMPROVEMENT
                        );
                        if let Some(ref pb) = pb {
                            pb.finish_and_clear();
                            println!(
                                "{:>12} graph optimization (no improvement)",
                                "\x1b[1;32mFinished\x1b[0m"
                            );
                        }
                        early_terminated = true;
                        break;
                    }
                }
            }
        }

        if !early_terminated && let Some(pb) = pb {
            pb.finish_and_clear();
            // Cargoスタイルの完了メッセージ
            println!("{:>12} graph optimization", "\x1b[1;32mFinished\x1b[0m");
        }

        debug!("BeamSearchGraphOptimizer: Beam search optimization complete");

        // 最良の候補を返す（beamは既にコスト順にソート済み）
        let best_graph = beam.into_iter().next().unwrap();

        (best_graph, history)
    }
}

impl<S, E> GraphOptimizer for BeamSearchGraphOptimizer<S, E>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
{
    fn optimize(&self, graph: Graph) -> Graph {
        // optimize_with_history()を呼び出して、グラフだけを返す
        let (optimized_graph, _history) = self.optimize_with_history(graph);
        optimized_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};
    use crate::opt::graph::{GraphSuggester, SimpleCostEstimator};

    // テスト用のダミーSuggester
    struct DummySuggester;

    impl GraphSuggester for DummySuggester {
        fn suggest(&self, _graph: &Graph) -> Vec<Graph> {
            // 何も提案しない
            vec![]
        }
    }

    #[test]
    fn test_beam_search_optimizer_no_candidates() {
        let suggester = DummySuggester;
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
            .with_beam_width(5)
            .with_max_steps(5)
            .with_progress(false);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        graph.output("a", a);

        let result = optimizer.optimize(graph);
        // 候補がないので元のグラフが返るはず
        assert_eq!(result.outputs().len(), 1);
    }

    #[test]
    fn test_output_order_independence() {
        // 出力順序が異なるグラフのDOT文字列が同じになることを確認

        // グラフ1: 出力順序 "out_a", "out_b"
        let mut graph1 = Graph::new();
        let x = graph1
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let y = graph1
            .input("y")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let result_a = x.clone() + y.clone();
        let result_b = x.clone() * y.clone();
        graph1.output("out_a", result_a);
        graph1.output("out_b", result_b);

        // グラフ2: 出力順序 "out_b", "out_a" (逆順)
        let mut graph2 = Graph::new();
        let x2 = graph2
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let y2 = graph2
            .input("y")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let result_b2 = x2.clone() * y2.clone();
        let result_a2 = x2.clone() + y2.clone();
        graph2.output("out_b", result_b2);
        graph2.output("out_a", result_a2);

        // 両方のグラフは同じDOT文字列を生成すべき
        let dot1 = graph1.to_dot();
        let dot2 = graph2.to_dot();

        assert_eq!(
            dot1, dot2,
            "Graphs with different output order should produce the same DOT signature"
        );
    }
}
