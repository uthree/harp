use crate::graph::Graph;
use crate::opt::graph::{GraphCostEstimator, GraphOptimizer, GraphSuggester};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;

/// ビームサーチグラフ最適化器
pub struct BeamSearchGraphOptimizer<S, E>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
{
    suggester: S,
    estimator: E,
    beam_width: usize,
    max_depth: usize,
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
            max_depth: 10,
            show_progress: true,
        }
    }

    /// ビーム幅を設定
    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    /// 最大深さを設定
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// プログレスバーの表示/非表示を設定
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }
}

impl<S, E> GraphOptimizer for BeamSearchGraphOptimizer<S, E>
where
    S: GraphSuggester,
    E: GraphCostEstimator,
{
    fn optimize(&self, graph: Graph) -> Graph {
        debug!("BeamSearchGraphOptimizer: Starting beam search optimization");

        let mut beam = vec![graph];

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_depth as u64);

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

        for depth in 0..self.max_depth {
            if let Some(ref pb) = pb {
                pb.set_message(format!("depth {}", depth + 1));
                pb.set_position(depth as u64);
            }

            let mut candidates = Vec::new();

            // 現在のビーム内の各候補から新しい候補を生成
            for graph in &beam {
                let new_candidates = self.suggester.suggest(graph);
                candidates.extend(new_candidates);
            }

            if candidates.is_empty() {
                debug!(
                    "BeamSearchGraphOptimizer: No more candidates at depth {}",
                    depth
                );
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_depth as u64);
                }
                break;
            }

            debug!(
                "BeamSearchGraphOptimizer: Found {} candidates at depth {}",
                candidates.len(),
                depth
            );

            // コストでソートして上位beam_width個を残す
            candidates.sort_by(|a, b| {
                self.estimator
                    .estimate(a)
                    .partial_cmp(&self.estimator.estimate(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            beam = candidates.into_iter().take(self.beam_width).collect();
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Complete");
        }

        debug!("BeamSearchGraphOptimizer: Beam search optimization complete");

        // 最良の候補を返す
        beam.into_iter()
            .min_by(|a, b| {
                self.estimator
                    .estimate(a)
                    .partial_cmp(&self.estimator.estimate(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
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
            .with_max_depth(5)
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
}
