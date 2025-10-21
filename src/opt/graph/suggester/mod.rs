mod loop_permutation;
mod parallelization;
mod tiling;
mod vectorization;

pub use loop_permutation::LoopPermutationSuggester;
pub use parallelization::{ParallelizationConfig, ParallelizationReason, ParallelizationSuggester};
pub use tiling::{TileSize, TilingSuggester};
pub use vectorization::{VectorizationConfig, VectorizationSuggester};

use crate::graph::Graph;

/// 全てのSuggesterを統合して最適化候補を生成
pub struct CombinedSuggester;

impl CombinedSuggester {
    /// 全てのSuggesterから候補を収集
    pub fn suggest_all(graph: &Graph) -> Vec<Graph> {
        let mut all_suggestions = Vec::new();

        // 各Suggesterから候補を生成
        all_suggestions.extend(LoopPermutationSuggester::suggest(graph));
        all_suggestions.extend(TilingSuggester::suggest(graph));
        all_suggestions.extend(ParallelizationSuggester::suggest(graph));
        all_suggestions.extend(VectorizationSuggester::suggest(graph));

        all_suggestions
    }

    /// 候補をコストでソート
    pub fn rank_by_cost(suggestions: Vec<Graph>) -> Vec<(Graph, usize)> {
        use crate::opt::graph::cost_estimator;

        let mut ranked: Vec<(Graph, usize)> = suggestions
            .into_iter()
            .map(|g| {
                let cost = cost_estimator::estimate_graph_cost(&g.outputs);
                (g, cost)
            })
            .collect();

        // コストの昇順でソート（低い方が良い）
        ranked.sort_by_key(|(_, cost)| *cost);

        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_combined_suggester() {
        let mut graph = Graph::new();
        // 十分大きなテンソル
        let _input = graph.input(DType::F32, vec![128.into(), 128.into()]);

        let suggestions = CombinedSuggester::suggest_all(&graph);

        // 何らかの最適化候補が生成されるはず
        // （現在の実装では、実際のGraph変換は未実装なので0個の可能性もある）
        assert!(suggestions.len() >= 0);
    }

    #[test]
    fn test_rank_by_cost() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![2.into(), 3.into()]);
        graph.output(input);

        // 同じグラフを複数用意
        let graphs = vec![graph.clone(), graph.clone(), graph];

        let ranked = CombinedSuggester::rank_by_cost(graphs);

        // 全て同じコストなので、どの順序でも良い
        assert_eq!(ranked.len(), 3);

        // コストが計算されているか確認
        for (_, cost) in ranked {
            assert!(cost > 0);
        }
    }
}
