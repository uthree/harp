use crate::graph::shape::Expr;
use crate::graph::{Graph, GraphNode};

/// タイリング最適化の提案を行う構造体
pub struct TilingSuggester {
    /// 使用可能なタイルサイズ
    pub tile_sizes: Vec<usize>,
}

impl Default for TilingSuggester {
    fn default() -> Self {
        Self {
            tile_sizes: vec![8, 16, 32, 64],
        }
    }
}

/// タイルサイズの候補
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSize {
    pub dim: usize,
    pub size: usize,
}

impl TilingSuggester {
    /// キャッシュ効率を向上させるタイリング提案を生成
    ///
    /// 戦略：
    /// 1. 大きな次元をタイルに分割
    /// 2. L1/L2キャッシュに収まるサイズを選択
    /// 3. 複数のタイルサイズ候補を生成
    pub fn suggest_internal(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();

        // 各出力ノードに対してタイリングを試みる
        for output in &graph.outputs {
            let tile_candidates = self.find_tiling_opportunities(output);

            for tile_config in tile_candidates {
                if let Some(tiled_graph) = Self::apply_tiling(graph, output, &tile_config) {
                    suggestions.push(tiled_graph);
                }
            }
        }

        suggestions
    }

    /// 互換性のためのstatic method
    pub fn suggest(graph: &Graph) -> Vec<Graph> {
        Self::default().suggest_internal(graph)
    }

    /// ノードのshapeからタイリング可能な次元を見つける
    fn find_tiling_opportunities(&self, node: &GraphNode) -> Vec<Vec<TileSize>> {
        let shape = node.view.shape();
        let mut opportunities = Vec::new();

        // 各次元について、タイリング候補を生成
        for (dim, size_expr) in shape.iter().enumerate() {
            if let Some(size) = Self::extract_constant_size(size_expr) {
                // サイズが十分大きい場合のみタイリングを検討
                if size >= 32 {
                    for &tile_size in &self.tile_sizes {
                        if size > tile_size * 2 {
                            // 単一次元のタイリング提案
                            opportunities.push(vec![TileSize {
                                dim,
                                size: tile_size,
                            }]);
                        }
                    }
                }
            }
        }

        // 2次元タイリング（行列演算など）
        if shape.len() >= 2 {
            let last_two_dims = &shape[shape.len() - 2..];
            if let (Some(dim0_size), Some(dim1_size)) = (
                Self::extract_constant_size(&last_two_dims[0]),
                Self::extract_constant_size(&last_two_dims[1]),
            ) {
                // 両方の次元が十分大きい場合、2D タイリングを提案
                if dim0_size >= 32 && dim1_size >= 32 {
                    for &tile_size in &self.tile_sizes {
                        if dim0_size > tile_size * 2 && dim1_size > tile_size * 2 {
                            opportunities.push(vec![
                                TileSize {
                                    dim: shape.len() - 2,
                                    size: tile_size,
                                },
                                TileSize {
                                    dim: shape.len() - 1,
                                    size: tile_size,
                                },
                            ]);
                        }
                    }
                }
            }
        }

        opportunities
    }

    /// Exprから定数値を抽出
    fn extract_constant_size(expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Const(val) if *val > 0 => Some(*val as usize),
            _ => None,
        }
    }

    /// タイリングをGraphに適用
    fn apply_tiling(_graph: &Graph, _node: &GraphNode, _tile_config: &[TileSize]) -> Option<Graph> {
        // TODO: 実際のタイリング適用ロジックを実装
        // 1. reshape を使って次元を分割 (N, M) -> (N/tile, tile, M/tile, tile)
        // 2. 必要に応じて pad を挿入
        // 3. LoopStrategy にタイリング情報を設定

        None
    }

    /// タイルサイズから推定されるキャッシュ使用量を計算
    pub fn estimate_cache_usage(tile_sizes: &[TileSize], element_size: usize) -> usize {
        let mut total_elements = 1;
        for tile in tile_sizes {
            total_elements *= tile.size;
        }
        total_elements * element_size
    }

    /// L1/L2キャッシュに収まるかチェック
    pub fn fits_in_l1_cache(cache_usage: usize) -> bool {
        // 典型的なL1キャッシュサイズ: 32KB
        cache_usage <= 32 * 1024
    }

    pub fn fits_in_l2_cache(cache_usage: usize) -> bool {
        // 典型的なL2キャッシュサイズ: 256KB
        cache_usage <= 256 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_extract_constant_size() {
        assert_eq!(
            TilingSuggester::extract_constant_size(&Expr::from(64)),
            Some(64)
        );
        assert_eq!(
            TilingSuggester::extract_constant_size(&Expr::Var("N".to_string())),
            None
        );
    }

    #[test]
    fn test_find_tiling_opportunities_large_matrix() {
        let mut graph = Graph::new();
        // 128x128 の大きな行列
        let node = graph.input(DType::F32, vec![128.into(), 128.into()]);

        let suggester = TilingSuggester::default();
        let opportunities = suggester.find_tiling_opportunities(&node);

        // 複数のタイリング候補が生成されるはず
        assert!(!opportunities.is_empty());

        // 2Dタイリングが含まれているか確認
        let has_2d_tiling = opportunities.iter().any(|config| config.len() == 2);
        assert!(has_2d_tiling);
    }

    #[test]
    fn test_find_tiling_opportunities_small_matrix() {
        let mut graph = Graph::new();
        // 8x8 の小さな行列
        let node = graph.input(DType::F32, vec![8.into(), 8.into()]);

        let suggester = TilingSuggester::default();
        let opportunities = suggester.find_tiling_opportunities(&node);

        // 小さすぎるのでタイリング候補なし
        assert_eq!(opportunities.len(), 0);
    }

    #[test]
    fn test_estimate_cache_usage() {
        let tiles = vec![TileSize { dim: 0, size: 8 }, TileSize { dim: 1, size: 8 }];

        // 8x8 * 4bytes (f32) = 256 bytes
        let usage = TilingSuggester::estimate_cache_usage(&tiles, 4);
        assert_eq!(usage, 256);
        assert!(TilingSuggester::fits_in_l1_cache(usage));
    }

    #[test]
    fn test_cache_fitting() {
        // L1に収まるサイズ
        assert!(TilingSuggester::fits_in_l1_cache(16 * 1024));

        // L1を超えるがL2には収まる
        assert!(!TilingSuggester::fits_in_l1_cache(64 * 1024));
        assert!(TilingSuggester::fits_in_l2_cache(64 * 1024));

        // L2も超える
        assert!(!TilingSuggester::fits_in_l2_cache(512 * 1024));
    }

    #[test]
    fn test_default_tile_sizes() {
        // デフォルトのタイルサイズが適切に設定されているか
        let suggester = TilingSuggester::default();
        assert!(suggester.tile_sizes.contains(&8));
        assert!(suggester.tile_sizes.contains(&16));
        assert!(suggester.tile_sizes.contains(&32));
        assert!(suggester.tile_sizes.contains(&64));
    }

    #[test]
    fn test_custom_tile_sizes() {
        let mut graph = Graph::new();
        // 128の次元
        let node = graph.input(DType::F32, vec![128.into()]);

        // カスタムタイルサイズで検証
        let custom_suggester = TilingSuggester {
            tile_sizes: vec![16, 32], // 8と64を除外
        };
        let opportunities = custom_suggester.find_tiling_opportunities(&node);

        // タイル候補が生成されることを確認
        assert!(!opportunities.is_empty());

        // 全ての候補が指定したサイズのみを使用していることを確認
        for config in &opportunities {
            for tile in config {
                assert!(
                    tile.size == 16 || tile.size == 32,
                    "Unexpected tile size: {}",
                    tile.size
                );
            }
        }
    }
}

// GraphSuggester trait implementation
use super::GraphSuggester;

impl GraphSuggester for TilingSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        self.suggest_internal(graph)
    }

    fn name(&self) -> &str {
        "Tiling"
    }

    fn priority(&self) -> usize {
        50
    }

    fn description(&self) -> &str {
        "Cache-friendly loop tiling"
    }
}
