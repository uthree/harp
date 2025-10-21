use crate::graph::{Graph, GraphNode, LoopStrategy};

/// 並列化戦略の提案を行う構造体
pub struct ParallelizationSuggester;

/// 並列化の候補
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelizationConfig {
    pub axes: Vec<usize>,
    pub reason: ParallelizationReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelizationReason {
    /// 大きな独立した次元
    LargeIndependentDimension,
    /// 外側のループ（データ並列性）
    OuterLoop,
    /// バッチ次元
    BatchDimension,
}

impl ParallelizationSuggester {
    /// 並列化可能な軸を提案
    ///
    /// 戦略：
    /// 1. データ依存性がない外側のループを並列化
    /// 2. 十分に大きなサイズの次元を優先
    /// 3. バッチ次元を優先的に並列化
    pub fn suggest(graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();

        // 各出力ノードに対して並列化戦略を生成
        for output in &graph.outputs {
            let configs = Self::find_parallelizable_axes(output);

            for config in configs {
                if let Some(parallelized_graph) =
                    Self::apply_parallelization(graph, output, &config)
                {
                    suggestions.push(parallelized_graph);
                }
            }
        }

        suggestions
    }

    /// ノードから並列化可能な軸を見つける
    fn find_parallelizable_axes(node: &GraphNode) -> Vec<ParallelizationConfig> {
        let shape = node.view.shape();
        let mut configs = Vec::new();

        if shape.is_empty() {
            return configs;
        }

        // 単一軸の並列化候補
        for (axis, size_expr) in shape.iter().enumerate() {
            if let Some(size) = Self::extract_constant_size(size_expr) {
                // 十分に大きい次元のみ並列化を検討
                if size >= 4 {
                    let reason = if axis == 0 {
                        ParallelizationReason::OuterLoop
                    } else {
                        ParallelizationReason::LargeIndependentDimension
                    };

                    configs.push(ParallelizationConfig {
                        axes: vec![axis],
                        reason,
                    });
                }
            }
        }

        // 複数軸の並列化（ネストされた並列化）
        // GPUの場合、2D/3Dグリッドが有効
        if shape.len() >= 2 {
            // 最外の2次元を並列化
            if let (Some(size0), Some(size1)) = (
                Self::extract_constant_size(&shape[0]),
                Self::extract_constant_size(&shape[1]),
            ) {
                if size0 >= 2 && size1 >= 2 {
                    configs.push(ParallelizationConfig {
                        axes: vec![0, 1],
                        reason: ParallelizationReason::BatchDimension,
                    });
                }
            }
        }

        configs
    }

    /// 定数サイズを抽出
    fn extract_constant_size(expr: &crate::graph::shape::Expr) -> Option<usize> {
        use crate::graph::shape::Expr;
        match expr {
            Expr::Const(val) if *val > 0 => Some(*val as usize),
            _ => None,
        }
    }

    /// 並列化戦略をGraphに適用
    fn apply_parallelization(
        graph: &Graph,
        node: &GraphNode,
        config: &ParallelizationConfig,
    ) -> Option<Graph> {
        let new_graph = graph.clone();

        // TODO: 実際のLoopStrategy設定を実装
        // ノードのstrategyフィールドにparallelizeを設定
        let _strategy = LoopStrategy {
            parallelize: config.axes.clone(),
            ..Default::default()
        };

        // 現時点では、Graphの変更は実装されていないため、
        // 単に同じグラフを返す（プレースホルダー）
        let _ = node;
        Some(new_graph)
    }

    /// 並列化によるスピードアップ比を推定
    pub fn estimate_speedup(config: &ParallelizationConfig, shape: &[usize]) -> f64 {
        let mut total_parallelism = 1.0;

        for &axis in &config.axes {
            if axis < shape.len() {
                total_parallelism *= shape[axis] as f64;
            }
        }

        // Amdahlの法則を簡易的に適用
        // 実際には並列化効率（通常70-90%）を考慮
        let efficiency = 0.85;
        total_parallelism * efficiency
    }

    /// 並列化によるオーバーヘッドを推定
    pub fn estimate_overhead(num_threads: usize) -> usize {
        // スレッド生成・同期のコスト
        // 典型的には数マイクロ秒〜数ミリ秒
        num_threads * 1000 // サイクル数で概算
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_find_parallelizable_axes_simple() {
        let mut graph = Graph::new();
        // バッチサイズ32, 特徴量64の入力
        let node = graph.input(DType::F32, vec![32.into(), 64.into()]);

        let configs = ParallelizationSuggester::find_parallelizable_axes(&node);

        // 少なくとも外側の次元（バッチ）を並列化可能
        assert!(!configs.is_empty());

        // バッチ次元（axis=0）の並列化が含まれている
        let has_batch_parallel = configs.iter().any(|c| c.axes.contains(&0));
        assert!(has_batch_parallel);
    }

    #[test]
    fn test_find_parallelizable_axes_2d() {
        let mut graph = Graph::new();
        // 128x128の行列
        let node = graph.input(DType::F32, vec![128.into(), 128.into()]);

        let configs = ParallelizationSuggester::find_parallelizable_axes(&node);

        // 2D並列化の候補が含まれているはず
        let has_2d_parallel = configs.iter().any(|c| c.axes.len() == 2);
        assert!(has_2d_parallel);
    }

    #[test]
    fn test_find_parallelizable_axes_small() {
        let mut graph = Graph::new();
        // 小さすぎる次元
        let node = graph.input(DType::F32, vec![2.into(), 3.into()]);

        let configs = ParallelizationSuggester::find_parallelizable_axes(&node);

        // 小さい次元でも一応並列化候補は生成される（サイズ>=2）
        assert!(!configs.is_empty());
    }

    #[test]
    fn test_estimate_speedup() {
        let config = ParallelizationConfig {
            axes: vec![0],
            reason: ParallelizationReason::BatchDimension,
        };

        // バッチサイズ32での並列化
        let speedup = ParallelizationSuggester::estimate_speedup(&config, &[32, 64]);

        // 効率85%と仮定すると、32 * 0.85 = 27.2倍
        assert!((speedup - 27.2).abs() < 0.1);
    }

    #[test]
    fn test_estimate_speedup_2d() {
        let config = ParallelizationConfig {
            axes: vec![0, 1],
            reason: ParallelizationReason::BatchDimension,
        };

        // 8x8での2D並列化
        let speedup = ParallelizationSuggester::estimate_speedup(&config, &[8, 8]);

        // 8 * 8 * 0.85 = 54.4倍
        assert!((speedup - 54.4).abs() < 0.1);
    }

    #[test]
    fn test_estimate_overhead() {
        let overhead = ParallelizationSuggester::estimate_overhead(4);
        // 4スレッド × 1000サイクル = 4000サイクル
        assert_eq!(overhead, 4000);
    }

    #[test]
    fn test_parallelization_reason() {
        let config1 = ParallelizationConfig {
            axes: vec![0],
            reason: ParallelizationReason::OuterLoop,
        };

        let config2 = ParallelizationConfig {
            axes: vec![1],
            reason: ParallelizationReason::LargeIndependentDimension,
        };

        // 異なる理由が適切に設定されているか
        assert_ne!(config1.reason, config2.reason);
    }
}
