use crate::graph::Graph;
use crate::opt::graph::cost_estimator;
use crate::opt::graph::suggester::CombinedSuggester;

/// ビームサーチによるGraph最適化
pub struct BeamSearchOptimizer {
    beam_width: usize,
    max_depth: usize,
    suggester: CombinedSuggester,
}

/// 最適化の候補（Graphとそのコスト）
#[derive(Clone)]
struct Candidate {
    graph: Graph,
    cost: usize,
    depth: usize,
}

impl BeamSearchOptimizer {
    /// デフォルト設定で最適化器を作成
    pub fn new() -> Self {
        Self {
            beam_width: 3, // デフォルトビーム幅
            max_depth: 5,  // デフォルト探索深さ
            suggester: CombinedSuggester::new(),
        }
    }

    /// ビーム幅を指定して最適化器を作成
    pub fn with_beam_width(beam_width: usize) -> Self {
        Self {
            beam_width,
            max_depth: 5,
            suggester: CombinedSuggester::new(),
        }
    }

    /// ビーム幅と探索深さを指定
    pub fn with_params(beam_width: usize, max_depth: usize) -> Self {
        Self {
            beam_width,
            max_depth,
            suggester: CombinedSuggester::new(),
        }
    }

    /// カスタムSuggesterを使用
    pub fn with_suggester(mut self, suggester: CombinedSuggester) -> Self {
        self.suggester = suggester;
        self
    }

    /// ビームサーチでGraphを最適化
    ///
    /// アルゴリズム：
    /// 1. 初期Graphをビームに追加
    /// 2. 各ステップで、ビーム内の各候補に対してSuggesterを適用
    /// 3. 生成された候補をコスト順にソート
    /// 4. 上位beam_width個を保持
    /// 5. max_depth到達またはコスト改善が停止したら終了
    pub fn optimize(&self, initial_graph: &Graph) -> Graph {
        let initial_cost = cost_estimator::estimate_graph_cost(&initial_graph.outputs);
        let mut beam = vec![Candidate {
            graph: initial_graph.clone(),
            cost: initial_cost,
            depth: 0,
        }];

        let mut best_candidate = beam[0].clone();
        let mut no_improvement_count = 0;

        for _iteration in 0..self.max_depth {
            let mut new_candidates = Vec::new();

            // 各候補に対してSuggesterを適用
            for candidate in &beam {
                // 最大深度チェック
                if candidate.depth >= self.max_depth {
                    continue;
                }

                // 全てのSuggesterから候補を生成
                let suggestions = self.suggester.suggest_all(&candidate.graph);

                for suggestion in suggestions {
                    let cost = cost_estimator::estimate_graph_cost(&suggestion.outputs);
                    new_candidates.push(Candidate {
                        graph: suggestion,
                        cost,
                        depth: candidate.depth + 1,
                    });
                }
            }

            // 新しい候補がない場合は終了
            if new_candidates.is_empty() {
                break;
            }

            // コストでソート
            new_candidates.sort_by_key(|c| c.cost);

            // 最良候補を更新
            if new_candidates[0].cost < best_candidate.cost {
                best_candidate = new_candidates[0].clone();
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
            }

            // 改善が見られない場合は早期終了
            if no_improvement_count >= 2 {
                break;
            }

            // 上位beam_width個を保持
            new_candidates.truncate(self.beam_width);
            beam = new_candidates;

            // ビームが空になったら終了
            if beam.is_empty() {
                break;
            }
        }

        best_candidate.graph
    }

    /// 貪欲法で最適化（ビーム幅1の特殊ケース）
    pub fn greedy_optimize(initial_graph: &Graph) -> Graph {
        let optimizer = Self::with_beam_width(1);
        optimizer.optimize(initial_graph)
    }
}

impl Default for BeamSearchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph最適化のユーティリティ関数
pub fn optimize_graph(graph: &Graph) -> Graph {
    let optimizer = BeamSearchOptimizer::new();
    optimizer.optimize(graph)
}

/// カスタムパラメータでGraph最適化
pub fn optimize_graph_with_params(graph: &Graph, beam_width: usize, max_depth: usize) -> Graph {
    let optimizer = BeamSearchOptimizer::with_params(beam_width, max_depth);
    optimizer.optimize(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_beam_search_optimizer_creation() {
        let optimizer = BeamSearchOptimizer::new();
        assert_eq!(optimizer.beam_width, 3);
        assert_eq!(optimizer.max_depth, 5);
    }

    #[test]
    fn test_beam_search_optimizer_with_beam_width() {
        let optimizer = BeamSearchOptimizer::with_beam_width(5);
        assert_eq!(optimizer.beam_width, 5);
        assert_eq!(optimizer.max_depth, 5);
    }

    #[test]
    fn test_beam_search_optimizer_with_params() {
        let optimizer = BeamSearchOptimizer::with_params(10, 3);
        assert_eq!(optimizer.beam_width, 10);
        assert_eq!(optimizer.max_depth, 3);
    }

    #[test]
    fn test_optimize_simple_graph() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![2.into(), 3.into()]);
        graph.output(input);

        let optimizer = BeamSearchOptimizer::new();
        let optimized = optimizer.optimize(&graph);

        // 最適化後のグラフが返される
        // （現在はSuggesterが実際のGraph変換を実装していないため、同じグラフが返る）
        assert_eq!(optimized.outputs.len(), 1);
    }

    #[test]
    fn test_greedy_optimize() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![4.into(), 5.into()]);
        graph.output(input);

        let optimized = BeamSearchOptimizer::greedy_optimize(&graph);

        assert_eq!(optimized.outputs.len(), 1);
    }

    #[test]
    fn test_optimize_graph_utility() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![8.into(), 8.into()]);
        graph.output(input);

        let optimized = optimize_graph(&graph);

        assert_eq!(optimized.outputs.len(), 1);
    }

    #[test]
    fn test_optimize_graph_with_custom_params() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![16.into(), 16.into()]);
        graph.output(input);

        let optimized = optimize_graph_with_params(&graph, 2, 3);

        assert_eq!(optimized.outputs.len(), 1);
    }

    #[test]
    fn test_early_termination_no_improvement() {
        let mut graph = Graph::new();
        // 単純なグラフ（最適化の余地が少ない）
        let input = graph.input(DType::F32, vec![2.into(), 2.into()]);
        graph.output(input);

        let optimizer = BeamSearchOptimizer::with_params(3, 10);
        let optimized = optimizer.optimize(&graph);

        // 改善がない場合でも、有効なグラフが返される
        assert_eq!(optimized.outputs.len(), 1);
    }

    #[test]
    fn test_empty_beam_handling() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![1.into(), 1.into()]);
        graph.output(input);

        let optimizer = BeamSearchOptimizer::with_beam_width(0); // ビーム幅0
        let optimized = optimizer.optimize(&graph);

        // ビームが空でも、元のグラフが返される
        assert_eq!(optimized.outputs.len(), 1);
    }
}
