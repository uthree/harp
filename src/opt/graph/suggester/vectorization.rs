use crate::graph::{Graph, GraphNode};

/// ベクトル化戦略の提案を行う構造体
pub struct VectorizationSuggester {
    /// 最大ベクトル幅の制限（None = 無制限）
    pub max_vector_width: Option<usize>,
}

impl Default for VectorizationSuggester {
    fn default() -> Self {
        Self {
            max_vector_width: Some(16), // AVX-512相当
        }
    }
}

/// ベクトル化の候補
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorizationConfig {
    pub axis: usize,
    pub vector_width: usize,
}

impl VectorizationSuggester {
    /// 一般的なベクトル幅（SIMD命令セット依存）
    /// - SSE: 128bit = 4xf32 or 2xf64
    /// - AVX: 256bit = 8xf32 or 4xf64
    /// - AVX-512: 512bit = 16xf32 or 8xf64
    const COMMON_VECTOR_WIDTHS: &'static [usize] = &[2, 4, 8, 16];

    /// SIMD命令を活用するベクトル化提案を生成
    ///
    /// 戦略：
    /// 1. 最内ループをベクトル化
    /// 2. ベクトル幅で割り切れる、または余りを処理可能
    /// 3. メモリアクセスが連続している
    pub fn suggest_internal(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();

        // 各出力ノードに対してベクトル化を試みる
        for output in &graph.outputs {
            let configs = Self::find_vectorizable_loops(output);

            for config in configs {
                // max_vector_widthでフィルタリング
                if let Some(max_width) = self.max_vector_width {
                    if config.vector_width > max_width {
                        continue;
                    }
                }

                if let Some(vectorized_graph) = Self::apply_vectorization(graph, output, &config) {
                    suggestions.push(vectorized_graph);
                }
            }
        }

        suggestions
    }

    /// 互換性のためのstatic method
    pub fn suggest(graph: &Graph) -> Vec<Graph> {
        Self::default().suggest_internal(graph)
    }

    /// ベクトル化可能なループを見つける
    fn find_vectorizable_loops(node: &GraphNode) -> Vec<VectorizationConfig> {
        let shape = node.view.shape();
        let mut configs = Vec::new();

        if shape.is_empty() {
            return configs;
        }

        // ストライドを分析して、最内ループ（stride=1）を特定
        let innermost_axis = Self::find_innermost_contiguous_axis(node);

        if let Some(axis) = innermost_axis {
            if let Some(size) = Self::extract_constant_size(&shape[axis]) {
                // 各ベクトル幅でベクトル化を試みる
                for &width in Self::COMMON_VECTOR_WIDTHS {
                    // サイズがベクトル幅以上であれば候補に追加
                    if size >= width {
                        configs.push(VectorizationConfig {
                            axis,
                            vector_width: width,
                        });
                    }
                }
            }
        }

        configs
    }

    /// stride=1の最内ループを見つける
    fn find_innermost_contiguous_axis(node: &GraphNode) -> Option<usize> {
        use crate::graph::shape::view::View;

        match &node.view {
            View::Linear { strides, .. } => {
                // stride=1の次元を探す
                for (i, stride) in strides.iter().enumerate() {
                    if stride.is_one() {
                        return Some(i);
                    }
                }
                None
            }
        }
    }

    /// 定数サイズを抽出
    fn extract_constant_size(expr: &crate::graph::shape::Expr) -> Option<usize> {
        use crate::graph::shape::Expr;
        match expr {
            Expr::Const(val) if *val > 0 => Some(*val as usize),
            _ => None,
        }
    }

    /// ベクトル化をGraphに適用
    fn apply_vectorization(
        graph: &Graph,
        node: &GraphNode,
        config: &VectorizationConfig,
    ) -> Option<Graph> {
        let mut new_graph = graph.clone();

        // LoopStrategyを作成
        let strategy = crate::graph::LoopStrategy {
            vectorize: Some((config.axis, config.vector_width)),
            unroll: None,
            parallelize: vec![],
            tile: vec![],
            use_shared_memory: false,
        };

        // ノードにstrategyを設定した新しいノードを作成
        let new_node = node.clone().with_strategy(strategy);

        // outputsの中で該当ノードを置き換え
        for output in &mut new_graph.outputs {
            if output.is_same_node(node) {
                *output = new_node.clone();
            }
        }

        Some(new_graph)
    }

    /// ベクトル化による性能向上を推定
    pub fn estimate_performance_gain(config: &VectorizationConfig, loop_size: usize) -> f64 {
        let vector_width = config.vector_width as f64;

        // ベクトル化により、理論上はベクトル幅倍の性能向上
        // ただし、実際には以下の要因で効率が下がる：
        // 1. メモリアライメント
        // 2. ループ端の処理（余り）
        // 3. 命令発行の制約

        let efficiency = if loop_size.is_multiple_of(config.vector_width) {
            // 割り切れる場合は高効率
            0.95
        } else {
            // 余りがある場合は効率低下
            0.75
        };

        vector_width * efficiency
    }

    /// ベクトル化が有効かどうかを判定
    pub fn is_beneficial(config: &VectorizationConfig, loop_size: usize) -> bool {
        // ループサイズがベクトル幅の2倍以上あれば有効
        loop_size >= config.vector_width * 2
    }

    /// アライメント要件をチェック
    pub fn check_alignment(base_address: usize, vector_width: usize) -> bool {
        // ベクトル化にはメモリアライメントが重要
        // 例: AVX (256bit = 32bytes) の場合、32バイトアライメントが必要
        let alignment_bytes = vector_width * 4; // f32の場合
        base_address.is_multiple_of(alignment_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_find_vectorizable_loops_contiguous() {
        let mut graph = Graph::new();
        // Contiguous配列: stride = [4, 1]
        let node = graph.input(DType::F32, vec![32.into(), 64.into()]);

        let configs = VectorizationSuggester::find_vectorizable_loops(&node);

        // 最内ループ（axis=1）のベクトル化候補が生成されるはず
        assert!(!configs.is_empty());

        // 全ての候補がaxis=1を対象にしているか
        assert!(configs.iter().all(|c| c.axis == 1));

        // 複数のベクトル幅候補がある
        assert!(configs.len() > 1);
    }

    #[test]
    fn test_find_innermost_contiguous_axis() {
        // Contiguous view
        let mut graph = Graph::new();
        let node = graph.input(DType::F32, vec![32.into(), 64.into()]);

        let axis = VectorizationSuggester::find_innermost_contiguous_axis(&node);

        // 最内ループはaxis=1（stride=1）
        assert_eq!(axis, Some(1));
    }

    #[test]
    fn test_find_innermost_contiguous_axis_permuted() {
        // Permuted view: [32, 64] -> permute([1, 0]) -> shape=[64, 32], strides = [1, 64]
        let mut graph = Graph::new();
        let node = graph
            .input(DType::F32, vec![32.into(), 64.into()])
            .permute(vec![1, 0]);

        let axis = VectorizationSuggester::find_innermost_contiguous_axis(&node);

        // permute後はaxis=0がstride=1
        assert_eq!(axis, Some(0));
    }

    #[test]
    fn test_estimate_performance_gain_perfect_fit() {
        let config = VectorizationConfig {
            axis: 0,
            vector_width: 8,
        };

        // ループサイズが8の倍数
        let gain = VectorizationSuggester::estimate_performance_gain(&config, 64);

        // 8倍 * 0.95 = 7.6倍
        assert!((gain - 7.6).abs() < 0.1);
    }

    #[test]
    fn test_estimate_performance_gain_with_remainder() {
        let config = VectorizationConfig {
            axis: 0,
            vector_width: 8,
        };

        // ループサイズが8の倍数でない
        let gain = VectorizationSuggester::estimate_performance_gain(&config, 63);

        // 8倍 * 0.75 = 6.0倍（効率低下）
        assert!((gain - 6.0).abs() < 0.1);
    }

    #[test]
    fn test_is_beneficial() {
        let config = VectorizationConfig {
            axis: 0,
            vector_width: 8,
        };

        // ループサイズが十分大きい
        assert!(VectorizationSuggester::is_beneficial(&config, 64));

        // ループサイズがギリギリ
        assert!(VectorizationSuggester::is_beneficial(&config, 16));

        // ループサイズが小さすぎる
        assert!(!VectorizationSuggester::is_beneficial(&config, 8));
        assert!(!VectorizationSuggester::is_beneficial(&config, 4));
    }

    #[test]
    fn test_check_alignment() {
        // 32バイトアライメント（AVX）
        assert!(VectorizationSuggester::check_alignment(0, 8));
        assert!(VectorizationSuggester::check_alignment(32, 8));
        assert!(VectorizationSuggester::check_alignment(64, 8));

        // アライメントされていない
        assert!(!VectorizationSuggester::check_alignment(16, 8));
        assert!(!VectorizationSuggester::check_alignment(8, 8));
    }

    #[test]
    fn test_common_vector_widths() {
        // よく使われるベクトル幅が定義されているか
        assert!(VectorizationSuggester::COMMON_VECTOR_WIDTHS.contains(&4)); // SSE
        assert!(VectorizationSuggester::COMMON_VECTOR_WIDTHS.contains(&8)); // AVX
        assert!(VectorizationSuggester::COMMON_VECTOR_WIDTHS.contains(&16)); // AVX-512
    }

    #[test]
    fn test_max_vector_width_filtering() {
        let mut graph = Graph::new();
        // 1024要素の配列
        let a = graph.input(DType::F32, vec![1024.into()]);
        let b = graph.input(DType::F32, vec![1024.into()]);
        let c = a + b;
        graph.output(c);

        // max_vector_width = 4 に制限
        let suggester_limited = VectorizationSuggester {
            max_vector_width: Some(4),
        };
        let suggestions_limited = suggester_limited.suggest_internal(&graph);

        // デフォルト（max_vector_width = 16）
        let suggester_default = VectorizationSuggester::default();
        let suggestions_default = suggester_default.suggest_internal(&graph);

        // 制限版の方が候補数が少ないはず（幅8, 16が除外される）
        assert!(
            suggestions_limited.len() < suggestions_default.len(),
            "Limited suggester should produce fewer suggestions. Limited: {}, Default: {}",
            suggestions_limited.len(),
            suggestions_default.len()
        );
    }
}

// GraphSuggester trait implementation
use super::GraphSuggester;

impl GraphSuggester for VectorizationSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        self.suggest_internal(graph)
    }

    fn name(&self) -> &str {
        "Vectorization"
    }

    fn priority(&self) -> usize {
        100 // ベクトル化は高優先度
    }

    fn description(&self) -> &str {
        "SIMD vectorization for innermost loops"
    }
}
