use crate::graph::Graph;
use crate::opt::graph::GraphSuggester;

/// ループのタイル化に相当するView変更を提案するSuggester
///
/// 注: 完全なタイル化を実現するにはreshape操作が必要です。
/// 現在のViewはreshapeをサポートしていないため、この実装は将来の拡張のための
/// スケルトンとなっています。
///
/// 将来的な実装方針:
/// 1. shape [N, M] を [N/tile, tile, M/tile, tile] に変換（reshape）
/// 2. permuteで [N/tile, M/tile, tile, tile] に並べ替え
/// 3. これにより内側ループがタイルサイズになり、時間的局所性が向上
pub struct TilingSuggester {
    /// 試行するタイルサイズの候補
    #[allow(dead_code)]
    tile_sizes: Vec<Vec<usize>>,
}

impl TilingSuggester {
    /// 新しいTilingSuggesterを作成
    pub fn new(tile_sizes: Vec<Vec<usize>>) -> Self {
        Self { tile_sizes }
    }

    /// デフォルトのタイルサイズを使用
    pub fn with_default_tile_sizes() -> Self {
        Self::new(vec![
            vec![32, 32], // 32x32 タイル
            vec![64, 64], // 64x64 タイル
            vec![16, 16], // 16x16 タイル
        ])
    }
}

impl Default for TilingSuggester {
    fn default() -> Self {
        Self::with_default_tile_sizes()
    }
}

impl GraphSuggester for TilingSuggester {
    fn suggest(&self, _graph: &Graph) -> Vec<Graph> {
        // TODO: reshape操作が実装されたら、タイル化を実装
        // 現在はView::reshapeが未実装なため、候補を生成しない
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_tiling_suggester_not_implemented() {
        let suggester = TilingSuggester::with_default_tile_sizes();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 64])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![64, 64])
            .build();

        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // 現在は未実装なので、候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }
}
