mod loop_permutation;
mod parallelization;
mod tiling;
mod vectorization;

pub use loop_permutation::LoopPermutationSuggester;
pub use parallelization::{ParallelizationConfig, ParallelizationReason, ParallelizationSuggester};
pub use tiling::{TileSize, TilingSuggester};
pub use vectorization::{VectorizationConfig, VectorizationSuggester};

use crate::graph::Graph;

/// Graph最適化の提案を生成するSuggesterの共通インターフェース
pub trait GraphSuggester {
    /// 最適化候補を生成
    fn suggest(&self, graph: &Graph) -> Vec<Graph>;

    /// Suggesterの名前（デバッグ用）
    fn name(&self) -> &str;

    /// 優先度（高いほど優先、デフォルト: 0）
    fn priority(&self) -> usize {
        0
    }

    /// このSuggesterを有効にするか（デフォルト: true）
    fn is_enabled(&self) -> bool {
        true
    }

    /// Suggesterの説明（ログやドキュメント生成用）
    fn description(&self) -> &str {
        ""
    }
}

/// 全てのSuggesterを統合して最適化候補を生成
pub struct CombinedSuggester {
    suggesters: Vec<Box<dyn GraphSuggester>>,
}

impl CombinedSuggester {
    /// デフォルトのSuggesterセットで初期化
    pub fn new() -> Self {
        Self {
            suggesters: vec![
                Box::new(VectorizationSuggester::default()),
                Box::new(ParallelizationSuggester::default()),
                Box::new(LoopPermutationSuggester),
                Box::new(TilingSuggester::default()),
            ],
        }
    }

    /// 空のCombinedSuggesterを作成
    pub fn empty() -> Self {
        Self {
            suggesters: Vec::new(),
        }
    }

    /// Suggesterを追加
    pub fn add_suggester(&mut self, suggester: Box<dyn GraphSuggester>) -> &mut Self {
        self.suggesters.push(suggester);
        self
    }

    /// Suggesterを削除（名前で指定）
    pub fn remove_suggester(&mut self, name: &str) -> &mut Self {
        self.suggesters.retain(|s| s.name() != name);
        self
    }

    /// 全てのSuggesterから候補を収集
    pub fn suggest_all(&self, graph: &Graph) -> Vec<Graph> {
        let mut all_suggestions = Vec::new();

        // 優先度順にソート
        let mut sorted_suggesters: Vec<_> = self.suggesters.iter().collect();
        sorted_suggesters.sort_by_key(|s| std::cmp::Reverse(s.priority()));

        for suggester in sorted_suggesters {
            if !suggester.is_enabled() {
                continue;
            }

            let suggestions = suggester.suggest(graph);
            all_suggestions.extend(suggestions);
        }

        all_suggestions
    }

    /// 候補をコストでランク付け
    pub fn rank_by_cost(&self, suggestions: Vec<Graph>) -> Vec<(Graph, usize)> {
        use crate::opt::graph::cost_estimator;

        let mut ranked: Vec<(Graph, usize)> = suggestions
            .into_iter()
            .map(|g| {
                let cost = cost_estimator::estimate_graph_cost(&g.outputs);
                (g, cost)
            })
            .collect();

        ranked.sort_by_key(|(_, cost)| *cost);
        ranked
    }

    /// Suggesterの一覧を取得
    pub fn list_suggesters(&self) -> Vec<(&str, &str, usize)> {
        self.suggesters
            .iter()
            .map(|s| (s.name(), s.description(), s.priority()))
            .collect()
    }

    /// ビルダーパターンで構築
    pub fn builder() -> CombinedSuggesterBuilder {
        CombinedSuggesterBuilder::new()
    }
}

impl Default for CombinedSuggester {
    fn default() -> Self {
        Self::new()
    }
}

/// CombinedSuggesterのビルダー
pub struct CombinedSuggesterBuilder {
    suggesters: Vec<Box<dyn GraphSuggester>>,
}

impl CombinedSuggesterBuilder {
    pub fn new() -> Self {
        Self {
            suggesters: Vec::new(),
        }
    }

    /// ベクトル化を追加
    pub fn with_vectorization(mut self) -> Self {
        self.suggesters
            .push(Box::new(VectorizationSuggester::default()));
        self
    }

    /// カスタム設定のベクトル化を追加
    pub fn with_vectorization_custom(mut self, max_width: usize) -> Self {
        self.suggesters.push(Box::new(VectorizationSuggester {
            max_vector_width: Some(max_width),
        }));
        self
    }

    /// 並列化を追加
    pub fn with_parallelization(mut self) -> Self {
        self.suggesters
            .push(Box::new(ParallelizationSuggester::default()));
        self
    }

    /// タイリングを追加
    pub fn with_tiling(mut self) -> Self {
        self.suggesters.push(Box::new(TilingSuggester::default()));
        self
    }

    /// ループ順序変更を追加
    pub fn with_loop_permutation(mut self) -> Self {
        self.suggesters.push(Box::new(LoopPermutationSuggester));
        self
    }

    /// カスタムSuggesterを追加
    pub fn with_custom(mut self, suggester: Box<dyn GraphSuggester>) -> Self {
        self.suggesters.push(suggester);
        self
    }

    /// ビルド
    pub fn build(self) -> CombinedSuggester {
        CombinedSuggester {
            suggesters: self.suggesters,
        }
    }
}

impl Default for CombinedSuggesterBuilder {
    fn default() -> Self {
        Self::new()
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

        let suggester = CombinedSuggester::new();
        let _suggestions = suggester.suggest_all(&graph);

        // 何らかの最適化候補が生成されるはず
        // （現在の実装では、実際のGraph変換は未実装なので0個の可能性もある）
        // 特に検証は不要（suggestが正常に動作することを確認）
    }

    #[test]
    fn test_rank_by_cost() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![2.into(), 3.into()]);
        graph.output(input);

        // 同じグラフを複数用意
        let graphs = vec![graph.clone(), graph.clone(), graph];

        let suggester = CombinedSuggester::new();
        let ranked = suggester.rank_by_cost(graphs);

        // 全て同じコストなので、どの順序でも良い
        assert_eq!(ranked.len(), 3);

        // コストが計算されているか確認
        for (_, cost) in ranked {
            assert!(cost > 0);
        }
    }
}
