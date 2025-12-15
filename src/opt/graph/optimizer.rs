use crate::graph::Graph;
use crate::opt::graph::{
    GraphCostEstimator, GraphOptimizer, GraphSuggester, OptimizationHistory, OptimizationSnapshot,
    SimpleCostEstimator, SuggestResult,
};

use super::selector::{GraphCostSelector, GraphSelector};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, trace};
use std::time::Instant;

/// ビームサーチグラフ最適化器
///
/// # 終了条件
///
/// 最適化は以下のいずれかの条件で終了します：
/// - 最大ステップ数(`max_steps`)に達した
/// - Suggesterから新しい提案がなくなった（これ以上最適化できない）
///
/// # 候補選択
///
/// `GraphSelector`により候補選択処理を抽象化しています。
/// デフォルトは`GraphCostSelector`（静的コストでソートして上位n件を選択）。
/// `with_selector()`で`GraphRuntimeSelector`などのカスタム選択器を設定可能。
///
/// # コスト推定
///
/// `GraphSelector`がCostEstimatorを内包しており、内部でコストを計算します。
pub struct BeamSearchGraphOptimizer<S, Sel = GraphCostSelector>
where
    S: GraphSuggester,
    Sel: GraphSelector,
{
    suggester: S,
    selector: Sel,
    beam_width: usize,
    max_steps: usize,
    show_progress: bool,
    collect_logs: bool,
    /// 早期終了の閾値（改善なしステップ数）
    ///
    /// Some(n): n回連続で改善がなければ終了
    /// None: 早期終了を無効化
    early_termination_threshold: Option<usize>,
}

impl<S> BeamSearchGraphOptimizer<S, GraphCostSelector>
where
    S: GraphSuggester,
{
    /// 新しいビームサーチ最適化器を作成
    ///
    /// デフォルトでは`GraphCostSelector`を使用します。
    pub fn new(suggester: S) -> Self {
        Self {
            suggester,
            selector: GraphCostSelector::new(),
            beam_width: 10,
            max_steps: 10000,
            show_progress: cfg!(debug_assertions),
            collect_logs: cfg!(debug_assertions),
            early_termination_threshold: Some(10), // デフォルト: 10ステップ改善なしで終了
        }
    }
}

impl<S, Sel> BeamSearchGraphOptimizer<S, Sel>
where
    S: GraphSuggester,
    Sel: GraphSelector,
{
    /// カスタム選択器を設定
    ///
    /// デフォルトの`GraphCostSelector`の代わりに、
    /// `GraphRuntimeSelector`などのカスタム選択器を使用できます。
    ///
    /// # Example
    ///
    /// ```ignore
    /// use harp::opt::{GraphRuntimeSelector, BeamSearchGraphOptimizer};
    /// use harp::backend::opencl::{OpenCLRenderer, OpenCLCompiler};
    ///
    /// let selector = GraphRuntimeSelector::new(
    ///     OpenCLRenderer::new(),
    ///     OpenCLCompiler::new(),
    ///     |sig| create_buffers(sig),
    /// );
    ///
    /// let optimizer = BeamSearchGraphOptimizer::new(suggester)
    ///     .with_selector(selector);
    /// ```
    pub fn with_selector<NewSel>(self, selector: NewSel) -> BeamSearchGraphOptimizer<S, NewSel>
    where
        NewSel: GraphSelector,
    {
        BeamSearchGraphOptimizer {
            suggester: self.suggester,
            selector,
            beam_width: self.beam_width,
            max_steps: self.max_steps,
            show_progress: self.show_progress,
            collect_logs: self.collect_logs,
            early_termination_threshold: self.early_termination_threshold,
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

    /// 早期終了の閾値を設定
    ///
    /// Some(n): n回連続で改善がなければ終了
    /// None: 早期終了を無効化
    ///
    /// デフォルト: Some(10)
    pub fn with_early_termination_threshold(mut self, threshold: Option<usize>) -> Self {
        self.early_termination_threshold = threshold;
        self
    }
}

impl<S, Sel> BeamSearchGraphOptimizer<S, Sel>
where
    S: GraphSuggester,
    Sel: GraphSelector,
{
    /// グラフを最適化して、グラフと最適化履歴を返す
    pub fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        use crate::opt::log_capture;

        // 内部でSimpleCostEstimatorを使用
        let static_estimator = SimpleCostEstimator::new();

        // 時間計測を開始
        let start_time = Instant::now();

        // ログキャプチャを開始（collect_logsが有効な場合のみ）
        if self.collect_logs {
            log_capture::start_capture();
        }

        info!(
            "Graph optimization started (beam_width={}, max_steps={})",
            self.beam_width, self.max_steps
        );

        let mut history = OptimizationHistory::new();
        let mut beam = vec![graph.clone()];

        // 初期状態を記録
        let initial_cost = static_estimator.estimate(&graph);
        let initial_outputs = graph.outputs().len();

        // 初期状態の入力・出力ノード情報をログに出力
        info!(
            "Initial graph: {} inputs, {} outputs, cost={:.2e}",
            graph.input_metas().len(),
            graph.outputs().len(),
            initial_cost
        );
        for (name, node) in graph.outputs() {
            let op_type = format!("{:?}", node.op);
            debug!("Initial output '{}': {:?}", name, op_type);
            // 依存グラフ全体をダンプ
            fn dump_node(
                node: &crate::graph::GraphNode,
                prefix: &str,
                visited: &mut std::collections::HashSet<*const crate::graph::GraphNodeData>,
            ) {
                let ptr = node.as_ptr();
                if visited.contains(&ptr) {
                    trace!("{}{:?} (already visited)", prefix, node.op);
                    return;
                }
                visited.insert(ptr);
                trace!("{}{:?} ({} srcs)", prefix, node.op, node.src.len());
                for src in &node.src {
                    dump_node(src, &format!("{}  ", prefix), visited);
                }
            }
            let mut visited = std::collections::HashSet::new();
            dump_node(&node, "  ", &mut visited);
        }

        let initial_logs = if self.collect_logs {
            log_capture::get_captured_logs()
        } else {
            Vec::new()
        };

        // これまでで最良の候補を保持（graphを移動する前に初期化）
        let mut global_best = graph.clone();

        history.add_snapshot(OptimizationSnapshot::with_candidates(
            0,
            graph,
            initial_cost,
            format!("Initial graph ({} outputs)", initial_outputs),
            initial_logs,
            0, // 初期状態では候補数は0
        ));

        // 初期ログをクリア（各ステップで新しいログのみを記録するため）
        if self.collect_logs {
            log_capture::clear_logs();
        }

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
        let mut best_cost = initial_cost;
        let mut no_improvement_count = 0;

        for step in 0..self.max_steps {
            if let Some(ref pb) = pb {
                pb.set_message(format!("step {}", step + 1));
                pb.set_position(step as u64);
            }

            let mut candidates: Vec<SuggestResult> = Vec::new();

            // 現在のビーム内の各候補から新しい候補を生成（Suggester名付き）
            for graph in &beam {
                let new_candidates = self.suggester.suggest_named(graph);
                candidates.extend(new_candidates);
            }

            if candidates.is_empty() {
                info!(
                    "No more candidates at step {} - optimization complete",
                    step
                );
                if let Some(ref pb) = pb {
                    pb.finish_and_clear();
                    let elapsed = start_time.elapsed();
                    let time_str = if elapsed.as_secs() > 0 {
                        format!("{:.2}s", elapsed.as_secs_f64())
                    } else {
                        format!("{}ms", elapsed.as_millis())
                    };
                    println!(
                        "{:>12} graph optimization in {} (no more candidates)",
                        "\x1b[1;32mFinished\x1b[0m", time_str
                    );
                }
                early_terminated = true;
                break;
            }

            // 候補数を記録
            let num_candidates = candidates.len();
            trace!("Found {} candidates at step {}", num_candidates, step);

            // 候補を準備（Selectorが内部でコストを計算）
            let candidates_for_selector: Vec<(Graph, String)> = candidates
                .into_iter()
                .map(|result| (result.graph, result.suggester_name))
                .collect();

            // Selectorで上位beam_width個を選択（Selectorが内部でコストを計算）
            let selected = self
                .selector
                .select(candidates_for_selector, self.beam_width);

            // (Graph, f32, String) の形式に変換
            let top_candidates: Vec<(Graph, f32, String)> = selected
                .into_iter()
                .map(|((graph, name), cost)| (graph, cost, name))
                .collect();

            beam = top_candidates.iter().map(|(g, _, _)| g.clone()).collect();

            // このステップの最良候補を記録（既に計算したコストを再利用）
            if let Some((best, cost, suggester_name)) = top_candidates.first() {
                let num_outputs = best.outputs().len();
                let num_inputs = best.input_metas().len();

                // 入力・出力ノード数をログに出力
                trace!(
                    "Step {} - {} inputs, {} outputs",
                    step + 1,
                    num_inputs,
                    num_outputs
                );

                // 出力ノードの演算タイプもログに出力
                for (name, node) in best.outputs() {
                    let op_type = format!("{:?}", node.op);
                    trace!("Step {} - Output '{}': {:?}", step + 1, name, op_type);
                }

                let step_logs = if self.collect_logs {
                    log_capture::get_captured_logs()
                } else {
                    Vec::new()
                };

                // スナップショットを記録（早期終了前に必ず記録する）
                history.add_snapshot(OptimizationSnapshot::with_suggester(
                    step + 1,
                    best.clone(),
                    *cost,
                    format!(
                        "[{}] Step {}: {} candidates, beam width {}, outputs: {}",
                        suggester_name,
                        step + 1,
                        num_candidates,
                        beam.len(),
                        num_outputs
                    ),
                    step_logs,
                    num_candidates,
                    suggester_name.clone(),
                ));

                // このステップのログをクリア（次のステップで新しいログのみを記録するため）
                if self.collect_logs {
                    log_capture::clear_logs();
                }

                // コストが改善されない場合はカウンターを増やす
                if *cost >= best_cost {
                    no_improvement_count += 1;

                    // 早期終了の判定
                    if let Some(threshold) = self.early_termination_threshold {
                        debug!(
                            "Step {}: no improvement (current={:.2e}, best={:.2e}, {}/{})",
                            step, cost, best_cost, no_improvement_count, threshold
                        );

                        // 連続で改善がない場合は早期終了
                        if no_improvement_count >= threshold {
                            info!(
                                "No cost improvement for {} steps - optimization complete",
                                threshold
                            );
                            if let Some(ref pb) = pb {
                                pb.finish_and_clear();
                                let elapsed = start_time.elapsed();
                                let time_str = if elapsed.as_secs() > 0 {
                                    format!("{:.2}s", elapsed.as_secs_f64())
                                } else {
                                    format!("{}ms", elapsed.as_millis())
                                };
                                println!(
                                    "{:>12} graph optimization in {} (converged)",
                                    "\x1b[1;32mFinished\x1b[0m", time_str
                                );
                            }
                            early_terminated = true;
                            break;
                        }
                    }
                } else {
                    // コストが改善された場合はカウンターをリセット
                    no_improvement_count = 0;
                    let improvement_pct = (best_cost - *cost) / best_cost * 100.0;
                    info!(
                        "Step {}: cost improved {:.2e} -> {:.2e} ({:+.1}%)",
                        step, best_cost, *cost, -improvement_pct
                    );
                    best_cost = *cost;
                    global_best = best.clone();
                }
            }
        }

        if !early_terminated && let Some(pb) = pb {
            pb.finish_and_clear();
            // Cargoスタイルの完了メッセージ
            let elapsed = start_time.elapsed();
            let time_str = if elapsed.as_secs() > 0 {
                format!("{:.2}s", elapsed.as_secs_f64())
            } else {
                format!("{}ms", elapsed.as_millis())
            };
            println!(
                "{:>12} graph optimization in {}",
                "\x1b[1;32mFinished\x1b[0m", time_str
            );
        }

        let final_cost = static_estimator.estimate(&global_best);
        let improvement_pct = if initial_cost > 0.0 {
            (initial_cost - final_cost) / initial_cost * 100.0
        } else {
            0.0
        };
        info!(
            "Graph optimization complete: {} steps, cost {:.2e} -> {:.2e} ({:+.1}%)",
            history.snapshots().len() - 1,
            initial_cost,
            final_cost,
            -improvement_pct
        );

        // 最終結果が履歴の最後と異なる場合（コスト改善がなかったステップが最後の場合）、
        // global_bestを最終スナップショットとして追加
        let last_snapshot_cost = history
            .snapshots()
            .last()
            .map(|s| s.cost)
            .unwrap_or(f32::MAX);
        if (final_cost - last_snapshot_cost).abs() > f32::EPSILON {
            let final_logs = if self.collect_logs {
                log_capture::get_captured_logs()
            } else {
                Vec::new()
            };
            let final_step = history.len();
            history.add_snapshot(OptimizationSnapshot::with_candidates(
                final_step,
                global_best.clone(),
                final_cost,
                format!(
                    "[Final] Best result (cost={:.2e}, improved {:.1}%)",
                    final_cost, improvement_pct
                ),
                final_logs,
                0,
            ));
        }

        // これまでで最良の候補を返す
        (global_best, history)
    }
}

impl<S, Sel> GraphOptimizer for BeamSearchGraphOptimizer<S, Sel>
where
    S: GraphSuggester,
    Sel: GraphSelector,
{
    fn optimize(&self, graph: Graph) -> Graph {
        // optimize_with_history()を呼び出して、グラフだけを返す
        let (optimized_graph, _history) = self.optimize_with_history(graph);
        optimized_graph
    }

    fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        // BeamSearchGraphOptimizer固有のメソッドを呼び出す
        BeamSearchGraphOptimizer::optimize_with_history(self, graph)
    }
}

/// コスト比較なしでサジェスチョンを適用する貪欲最適化器
///
/// サブグラフインライン化など、コストが増加しても適用すべき変換に使用します。
/// 各ステップで最初のサジェスチョンを常に適用し、サジェスチョンがなくなるまで繰り返します。
///
/// # Example
///
/// ```
/// use harp::opt::graph::{GreedyGraphOptimizer, GraphOptimizer, SubgraphInliningSuggester};
/// use harp::graph::{Graph, DType};
///
/// let suggester = SubgraphInliningSuggester::new();
/// let optimizer = GreedyGraphOptimizer::new(suggester)
///     .with_max_steps(1000);
///
/// let mut graph = Graph::new();
/// let a = graph.input("a", DType::F32, vec![4]);
/// graph.output("out", a);
/// let optimized = optimizer.optimize(graph);
/// ```
pub struct GreedyGraphOptimizer<S>
where
    S: GraphSuggester,
{
    suggester: S,
    max_steps: usize,
    name: Option<String>,
}

impl<S> GreedyGraphOptimizer<S>
where
    S: GraphSuggester,
{
    /// 新しい貪欲最適化器を作成
    pub fn new(suggester: S) -> Self {
        Self {
            suggester,
            max_steps: 10000,
            name: None,
        }
    }

    /// 最大ステップ数を設定
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// 名前を設定
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 最適化を実行（履歴付き）
    pub fn optimize_with_history(&self, mut graph: Graph) -> (Graph, OptimizationHistory) {
        let mut history = OptimizationHistory::new();
        let estimator = SimpleCostEstimator::new();

        // 初期状態を記録
        let initial_cost = estimator.estimate(&graph);
        history.add_snapshot(OptimizationSnapshot::new(
            0,
            graph.clone(),
            initial_cost,
            "Initial graph".to_string(),
        ));

        info!(
            "GreedyGraphOptimizer: starting (max_steps={})",
            self.max_steps
        );

        for step in 0..self.max_steps {
            // サジェスチョンを取得
            let suggestions = self.suggester.suggest(&graph);

            if suggestions.is_empty() {
                info!("GreedyGraphOptimizer: no more suggestions at step {}", step);
                break;
            }

            // 最初のサジェスチョンを適用（コスト比較なし）
            let new_graph = suggestions.into_iter().next().unwrap();
            let cost = estimator.estimate(&new_graph);

            debug!(
                "GreedyGraphOptimizer: step {} - applied suggestion (cost: {:.2e})",
                step, cost
            );

            history.add_snapshot(OptimizationSnapshot::new(
                step + 1,
                new_graph.clone(),
                cost,
                format!("Step {} - greedy apply", step + 1),
            ));

            graph = new_graph;
        }

        let final_cost = estimator.estimate(&graph);
        info!(
            "GreedyGraphOptimizer: complete (final cost: {:.2e})",
            final_cost
        );

        (graph, history)
    }
}

impl<S> GraphOptimizer for GreedyGraphOptimizer<S>
where
    S: GraphSuggester,
{
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn optimize(&self, graph: Graph) -> Graph {
        let (optimized, _) = self.optimize_with_history(graph);
        optimized
    }

    fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        GreedyGraphOptimizer::optimize_with_history(self, graph)
    }
}

/// 複数のGraphOptimizerを順番に適用するチェーンオプティマイザ
///
/// 各オプティマイザはフェーズ名を持ち、履歴には各フェーズの名前がプレフィックスとして付与されます。
///
/// # Example
///
/// ```
/// use harp::opt::graph::{ChainedGraphOptimizer, GraphOptimizer};
/// use harp::backend::IdentityOptimizer;
/// use harp::graph::{Graph, DType};
///
/// let lowering_optimizer = IdentityOptimizer::new("lowering");
/// let fusion_optimizer = IdentityOptimizer::new("fusion");
///
/// let chain = ChainedGraphOptimizer::new()
///     .add_phase("Lowering", lowering_optimizer)
///     .add_phase("Fusion", fusion_optimizer);
///
/// let mut graph = Graph::new();
/// let a = graph.input("a", DType::F32, vec![4]);
/// graph.output("out", a);
///
/// let (optimized, history) = chain.optimize_with_history(graph);
/// ```
pub struct ChainedGraphOptimizer {
    /// (フェーズ名, オプティマイザ) のリスト
    phases: Vec<(String, Box<dyn GraphOptimizer>)>,
}

impl ChainedGraphOptimizer {
    /// 新しいチェーンオプティマイザを作成
    pub fn new() -> Self {
        Self { phases: Vec::new() }
    }

    /// フェーズを追加
    ///
    /// # Arguments
    /// * `name` - フェーズ名（履歴のプレフィックスとして使用）
    /// * `optimizer` - このフェーズで使用するオプティマイザ
    pub fn add_phase<O: GraphOptimizer + 'static>(
        mut self,
        name: impl Into<String>,
        optimizer: O,
    ) -> Self {
        self.phases.push((name.into(), Box::new(optimizer)));
        self
    }

    /// Boxedオプティマイザでフェーズを追加
    ///
    /// # Arguments
    /// * `name` - フェーズ名（履歴のプレフィックスとして使用）
    /// * `optimizer` - このフェーズで使用するオプティマイザ（Box化済み）
    pub fn add_phase_boxed(
        mut self,
        name: impl Into<String>,
        optimizer: Box<dyn GraphOptimizer>,
    ) -> Self {
        self.phases.push((name.into(), optimizer));
        self
    }

    /// フェーズ数を取得
    pub fn len(&self) -> usize {
        self.phases.len()
    }

    /// フェーズが空かどうか
    pub fn is_empty(&self) -> bool {
        self.phases.is_empty()
    }

    /// 他のオプティマイザをチェーンに追加
    ///
    /// 既存のチェーンに新しいフェーズを追加します。
    /// オプティマイザに`with_name()`で名前が設定されている場合はその名前を使用し、
    /// 設定されていない場合は "Phase N" と自動命名されます。
    ///
    /// # Example
    ///
    /// ```
    /// use harp::opt::graph::{ChainedGraphOptimizer, GraphOptimizer};
    /// use harp::backend::IdentityOptimizer;
    ///
    /// let optimizer1 = IdentityOptimizer::new("opt1");
    /// let optimizer2 = IdentityOptimizer::new("opt2");
    /// let optimizer3 = IdentityOptimizer::new("opt3");
    ///
    /// // 名前付きでチェーン
    /// let chained = ChainedGraphOptimizer::new()
    ///     .add_phase("Phase 1", optimizer1)
    ///     .chain(optimizer2.with_name("Fusion"))
    ///     .chain(optimizer3.with_name("Finalize"));
    ///
    /// // 名前なしでチェーン（自動命名）
    /// let optimizer1 = IdentityOptimizer::new("opt1");
    /// let optimizer2 = IdentityOptimizer::new("opt2");
    /// let optimizer3 = IdentityOptimizer::new("opt3");
    /// let chained = optimizer1.chain(optimizer2).chain(optimizer3);
    /// ```
    pub fn chain<O: GraphOptimizer + 'static>(mut self, other: O) -> Self {
        let phase_num = self.phases.len() + 1;
        let name = other
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Phase {}", phase_num));
        self.phases.push((name, Box::new(other)));
        self
    }
}

impl Default for ChainedGraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// 名前付きオプティマイザのラッパー
///
/// `GraphOptimizer::with_name()`で作成され、チェーン時にフェーズ名として使用されます。
///
/// # Example
///
/// ```
/// use harp::opt::graph::GraphOptimizer;
/// use harp::backend::IdentityOptimizer;
///
/// let optimizer = IdentityOptimizer::new("opt");
/// let other = IdentityOptimizer::new("other");
///
/// let named = optimizer.with_name("Preparation");
/// assert_eq!(named.name(), Some("Preparation"));
///
/// let chained = named.chain(other.with_name("Lowering"));
/// ```
pub struct NamedOptimizer<O: GraphOptimizer> {
    inner: O,
    name: String,
}

impl<O: GraphOptimizer> NamedOptimizer<O> {
    /// 新しいNamedOptimizerを作成
    pub fn new(inner: O, name: impl Into<String>) -> Self {
        Self {
            inner,
            name: name.into(),
        }
    }

    /// 内部のオプティマイザへの参照を取得
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// 内部のオプティマイザを取り出す
    pub fn into_inner(self) -> O {
        self.inner
    }
}

impl<O: GraphOptimizer> GraphOptimizer for NamedOptimizer<O> {
    fn name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn optimize(&self, graph: Graph) -> Graph {
        self.inner.optimize(graph)
    }

    fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        self.inner.optimize_with_history(graph)
    }
}

impl GraphOptimizer for ChainedGraphOptimizer {
    fn optimize(&self, graph: Graph) -> Graph {
        let (optimized, _) = self.optimize_with_history(graph);
        optimized
    }

    fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
        let mut current_graph = graph;
        let mut combined_history = OptimizationHistory::new();

        for (phase_name, optimizer) in &self.phases {
            info!("Starting phase: {}", phase_name);

            let (optimized, phase_history) = optimizer.optimize_with_history(current_graph);

            // フェーズの履歴を結合
            combined_history.extend_with_phase(phase_history, phase_name);

            current_graph = optimized;

            info!("Completed phase: {}", phase_name);
        }

        (current_graph, combined_history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};
    use crate::opt::graph::GraphSuggester;

    // テスト用のダミーSuggester
    struct DummySuggester;

    impl GraphSuggester for DummySuggester {
        fn name(&self) -> &'static str {
            "Dummy"
        }

        fn suggest(&self, _graph: &Graph) -> Vec<Graph> {
            // 何も提案しない
            vec![]
        }
    }

    #[test]
    fn test_beam_search_optimizer_no_candidates() {
        let suggester = DummySuggester;

        let optimizer = BeamSearchGraphOptimizer::new(suggester)
            .with_beam_width(5)
            .with_max_steps(5)
            .with_progress(false);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("a", a);

        let result = optimizer.optimize(graph);
        // 候補がないので元のグラフが返るはず
        assert_eq!(result.outputs().len(), 1);
    }

    // Note: test_output_order_independence は複数出力が
    // 現在サポートされていないため削除されました。

    // 履歴を記録するテスト用Optimizer
    struct HistoryRecordingOptimizer {
        phase_id: usize,
    }

    impl GraphOptimizer for HistoryRecordingOptimizer {
        fn optimize(&self, graph: Graph) -> Graph {
            graph
        }

        fn optimize_with_history(&self, graph: Graph) -> (Graph, OptimizationHistory) {
            let mut history = OptimizationHistory::new();
            let cost = 100.0 / (self.phase_id + 1) as f32;
            history.add_snapshot(OptimizationSnapshot::new(
                0,
                graph.clone(),
                cost,
                format!("Phase {} step 0", self.phase_id),
            ));
            history.add_snapshot(OptimizationSnapshot::with_suggester(
                1,
                graph.clone(),
                cost * 0.9,
                format!("Phase {} step 1", self.phase_id),
                vec![format!("Log from phase {}", self.phase_id)],
                5,
                format!("PhaseOptimizer{}", self.phase_id),
            ));
            (graph, history)
        }
    }

    #[test]
    fn test_chained_optimizer_empty() {
        let chain = ChainedGraphOptimizer::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, history) = chain.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);
        assert!(history.is_empty());
    }

    #[test]
    fn test_chained_optimizer_single_phase() {
        let chain = ChainedGraphOptimizer::new()
            .add_phase("Phase1", HistoryRecordingOptimizer { phase_id: 1 });

        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, history) = chain.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);

        // 履歴には2つのスナップショットがあるはず
        assert_eq!(history.len(), 2);

        // フェーズ名がプレフィックスとして付いているか確認
        let snapshots = history.snapshots();
        assert!(snapshots[0].description.contains("Phase1"));
        assert!(snapshots[1].description.contains("Phase1"));
    }

    #[test]
    fn test_chained_optimizer_multiple_phases() {
        let chain = ChainedGraphOptimizer::new()
            .add_phase("Lowering", HistoryRecordingOptimizer { phase_id: 1 })
            .add_phase("Fusion", HistoryRecordingOptimizer { phase_id: 2 })
            .add_phase("Finalize", HistoryRecordingOptimizer { phase_id: 3 });

        assert_eq!(chain.len(), 3);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, history) = chain.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);

        // 各フェーズから2スナップショット、ただし2番目以降の初期スナップショットはスキップ
        // Phase1: 2 snapshots (step 0, step 1)
        // Phase2: 1 snapshot (step 0 skipped, step 1)
        // Phase3: 1 snapshot (step 0 skipped, step 1)
        // 合計: 4 snapshots
        let snapshots = history.snapshots();
        assert_eq!(snapshots.len(), 4);

        // ステップ番号が連続しているか確認
        assert_eq!(snapshots[0].step, 0);
        assert_eq!(snapshots[1].step, 1);
        assert_eq!(snapshots[2].step, 2);
        assert_eq!(snapshots[3].step, 3);

        // 各フェーズ名が正しく付与されているか
        assert!(snapshots[0].description.contains("Lowering"));
        assert!(snapshots[1].description.contains("Lowering"));
        assert!(snapshots[2].description.contains("Fusion"));
        assert!(snapshots[3].description.contains("Finalize"));
    }

    #[test]
    fn test_chained_optimizer_with_beam_search() {
        let suggester = DummySuggester;

        let beam_optimizer = BeamSearchGraphOptimizer::new(suggester)
            .with_beam_width(5)
            .with_max_steps(5)
            .with_progress(false)
            .with_collect_logs(false);

        let chain = ChainedGraphOptimizer::new().add_phase("BeamSearch", beam_optimizer);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, history) = chain.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);

        // BeamSearchOptimizerからの履歴が含まれているはず
        assert!(!history.is_empty());
    }

    #[test]
    fn test_chain_method_from_optimizer() {
        use crate::opt::graph::GraphOptimizer;

        // GraphOptimizerのchainメソッドを使って2つのオプティマイザを結合
        let opt1 = HistoryRecordingOptimizer { phase_id: 1 };
        let opt2 = HistoryRecordingOptimizer { phase_id: 2 };

        let chained = opt1.chain(opt2);
        assert_eq!(chained.len(), 2);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, history) = chained.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);

        // 両方のフェーズの履歴が含まれているはず
        let snapshots = history.snapshots();
        assert!(snapshots.len() >= 2);

        // フェーズ名が "Phase 1", "Phase 2" になっているはず
        assert!(snapshots[0].description.contains("Phase 1"));
    }

    #[test]
    fn test_chain_method_multiple() {
        use crate::opt::graph::GraphOptimizer;

        // 3つのオプティマイザをメソッドチェーンで結合
        let opt1 = HistoryRecordingOptimizer { phase_id: 1 };
        let opt2 = HistoryRecordingOptimizer { phase_id: 2 };
        let opt3 = HistoryRecordingOptimizer { phase_id: 3 };

        // opt1.chain(opt2) で ChainedGraphOptimizer が返り、
        // その .chain(opt3) で既存のチェーンに追加される
        let chained = opt1.chain(opt2).chain(opt3);
        assert_eq!(chained.len(), 3);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, history) = chained.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);

        // すべてのフェーズの履歴が含まれているはず
        let snapshots = history.snapshots();
        assert!(snapshots.len() >= 3);
    }

    #[test]
    fn test_with_name_method() {
        use crate::opt::graph::GraphOptimizer;

        let opt1 = HistoryRecordingOptimizer { phase_id: 1 };
        let opt2 = HistoryRecordingOptimizer { phase_id: 2 };
        let opt3 = HistoryRecordingOptimizer { phase_id: 3 };

        // with_name()で名前を付けてからchain()でチェーン
        let chained = opt1
            .with_name("Lowering")
            .chain(opt2.with_name("Fusion"))
            .chain(opt3.with_name("Finalize"));

        assert_eq!(chained.len(), 3);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, history) = chained.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);

        // 指定したフェーズ名が使われているはず
        let snapshots = history.snapshots();
        assert!(snapshots[0].description.contains("Lowering"));
        assert!(snapshots[2].description.contains("Fusion"));
        assert!(snapshots[3].description.contains("Finalize"));
    }

    #[test]
    fn test_named_optimizer() {
        use crate::opt::graph::GraphOptimizer;

        let opt = HistoryRecordingOptimizer { phase_id: 1 };

        // 名前なしの場合
        assert!(opt.name().is_none());

        // 名前を付けた場合
        let named = opt.with_name("MyOptimizer");
        assert_eq!(named.name(), Some("MyOptimizer"));

        // 最適化は元のオプティマイザと同じ動作
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("out", a);

        let (result, _history) = named.optimize_with_history(graph);
        assert_eq!(result.outputs().len(), 1);
    }
}
