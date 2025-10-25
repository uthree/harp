//! Lowering戦略の最適化
//!
//! 複数の戦略候補を試して、ASTコストが最小になる戦略を選択します。

use crate::ast::AstNode;
use crate::graph::shape::Expr;
use crate::graph::{GraphNode, GraphOp, LoopStrategy};
use crate::lowerer::config::LoweringConfig;
use crate::lowerer::recursive::RecursiveLowerer;
use crate::opt::ast::{CostEstimator, OperationCostEstimator};

/// Lowering戦略を探索して最適なASTを選択
pub struct LoweringOptimizer {
    /// 設定パラメータ
    config: LoweringConfig,

    /// コスト推定器
    cost_estimator: OperationCostEstimator,
}

impl LoweringOptimizer {
    /// 新しいLoweringOptimizerを作成
    pub fn new(config: LoweringConfig) -> Self {
        Self {
            config,
            cost_estimator: OperationCostEstimator,
        }
    }

    /// デフォルト設定でLoweringOptimizerを作成
    pub fn with_default_config() -> Self {
        Self::new(LoweringConfig::default())
    }

    /// 単一ノードに対して戦略を最適化
    ///
    /// 複数の戦略候補を生成し、それぞれでloweringを試して、
    /// 最小コストの戦略とASTを返します。
    ///
    /// # Arguments
    ///
    /// * `node` - 最適化するGraphNode
    ///
    /// # Returns
    ///
    /// (最適化されたノード, AST, コスト)のタプル
    pub fn optimize_node(&self, node: &GraphNode) -> (GraphNode, AstNode, f32) {
        // 1. 候補戦略を生成
        let strategies = self.generate_strategies(node);

        // 2. 各戦略でloweringを試す
        let mut candidates = Vec::new();
        for strategy in strategies {
            // ノードに戦略を設定したバリアントを作成
            let variant = node.clone().with_strategy(strategy);

            // lowering実行
            let mut lowerer = RecursiveLowerer::new();
            let ast = lowerer.lower_node(&variant);

            // コストを評価
            let cost = self.cost_estimator.estimate_cost(&ast);

            candidates.push((variant, ast, cost));
        }

        // 3. 最小コストを選択
        candidates
            .into_iter()
            .min_by(|(_, _, cost1), (_, _, cost2)| {
                cost1
                    .partial_cmp(cost2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("At least one strategy candidate should exist")
    }

    /// ノードに適用可能な戦略候補を生成
    ///
    /// configの設定に基づいて候補を生成します。
    ///
    /// # Arguments
    ///
    /// * `node` - 戦略を生成するGraphNode
    ///
    /// # Returns
    ///
    /// LoopStrategy候補のベクトル
    fn generate_strategies(&self, node: &GraphNode) -> Vec<LoopStrategy> {
        let mut strategies = Vec::new();

        // 戦略なし（ベースライン）
        strategies.push(LoopStrategy::default());

        // 最適化が無効なら基本戦略のみ返す
        if !self.config.enable_optimization {
            return strategies;
        }

        match &node.op {
            GraphOp::Elementwise(_) | GraphOp::FusedElementwise(_, _) => {
                self.generate_elementwise_strategies(node, &mut strategies);
            }

            GraphOp::Reduce(_, _, _) | GraphOp::FusedReduce(_, _, _) => {
                self.generate_reduce_strategies(node, &mut strategies);
            }

            GraphOp::Cumulative(_, _, _) | GraphOp::FusedElementwiseCumulative(_, _, _, _) => {
                self.generate_cumulative_strategies(node, &mut strategies);
            }

            _ => {
                // その他の演算は戦略なし（デフォルトのみ）
            }
        }

        // ビーム幅で候補数を制限
        if strategies.len() > self.config.beam_width {
            strategies.truncate(self.config.beam_width);
        }

        strategies
    }

    /// Elementwise演算の戦略候補を生成
    fn generate_elementwise_strategies(
        &self,
        node: &GraphNode,
        strategies: &mut Vec<LoopStrategy>,
    ) {
        let innermost_axis = node.view.shape().len().saturating_sub(1);

        // ベクトル化の候補（configから取得）
        if self.config.enable_vectorization {
            for &width in &self.config.vectorize_widths {
                if self.can_vectorize(node, innermost_axis, width) {
                    strategies.push(LoopStrategy {
                        vectorize: Some((innermost_axis, width)),
                        ..Default::default()
                    });
                }
            }
        }

        // 並列化の候補
        if self.config.enable_parallelization && node.view.shape().len() > 1 {
            strategies.push(LoopStrategy {
                parallelize: vec![0], // 最外ループ
                ..Default::default()
            });

            // ベクトル化 + 並列化
            if self.config.enable_vectorization {
                for &width in &self.config.vectorize_widths {
                    if self.can_vectorize(node, innermost_axis, width) {
                        strategies.push(LoopStrategy {
                            vectorize: Some((innermost_axis, width)),
                            parallelize: vec![0],
                            ..Default::default()
                        });
                    }
                }
            }
        }

        // タイリングの候補（configから取得）
        if self.config.enable_tiling && node.view.shape().len() >= 2 {
            for &tile_size in &self.config.tile_sizes {
                strategies.push(LoopStrategy {
                    tile: vec![(0, tile_size), (1, tile_size)],
                    ..Default::default()
                });
            }
        }

        // アンロールの候補
        if !node.view.shape().is_empty() {
            for &unroll_factor in &self.config.unroll_factors {
                if self.can_unroll(node, innermost_axis, unroll_factor) {
                    strategies.push(LoopStrategy {
                        unroll: Some((innermost_axis, unroll_factor)),
                        ..Default::default()
                    });
                }
            }
        }
    }

    /// Reduce演算の戦略候補を生成
    fn generate_reduce_strategies(&self, node: &GraphNode, strategies: &mut Vec<LoopStrategy>) {
        // 縮約演算の候補
        // 並列化（最外ループのみ）
        if self.config.enable_parallelization && node.view.shape().len() > 1 {
            strategies.push(LoopStrategy {
                parallelize: vec![0],
                ..Default::default()
            });
        }
    }

    /// Cumulative演算の戦略候補を生成
    fn generate_cumulative_strategies(&self, node: &GraphNode, strategies: &mut Vec<LoopStrategy>) {
        // Cumulative演算は並列化が難しいため、基本的な最適化のみ
        // 並列化しない次元でアンロール
        if node.view.shape().len() > 1 {
            for &unroll_factor in &self.config.unroll_factors {
                if self.can_unroll(node, 0, unroll_factor) {
                    strategies.push(LoopStrategy {
                        unroll: Some((0, unroll_factor)),
                        ..Default::default()
                    });
                }
            }
        }
    }

    /// ベクトル化が可能かチェック
    fn can_vectorize(&self, node: &GraphNode, axis: usize, width: usize) -> bool {
        if axis >= node.view.shape().len() {
            return false;
        }

        // shape[axis]がwidthで割り切れるかチェック
        match &node.view.shape()[axis] {
            Expr::Const(n) => (*n as usize).is_multiple_of(width),
            _ => true, // 動的サイズの場合は試してみる
        }
    }

    /// アンロールが可能かチェック
    fn can_unroll(&self, node: &GraphNode, axis: usize, factor: usize) -> bool {
        if axis >= node.view.shape().len() {
            return false;
        }

        // shape[axis]がfactorで割り切れるかチェック
        match &node.view.shape()[axis] {
            Expr::Const(n) => (*n as usize).is_multiple_of(factor),
            _ => true, // 動的サイズの場合は試してみる
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::Graph;

    #[test]
    fn test_new_optimizer() {
        let optimizer = LoweringOptimizer::with_default_config();
        assert!(optimizer.config.enable_optimization);
    }

    #[test]
    fn test_generate_strategies_no_optimization() {
        let config = LoweringConfig::no_optimization();
        let optimizer = LoweringOptimizer::new(config);

        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);
        let result = -input;

        let strategies = optimizer.generate_strategies(&result);

        // 最適化無効の場合、デフォルト戦略のみ
        assert_eq!(strategies.len(), 1);
        assert_eq!(strategies[0], LoopStrategy::default());
    }

    #[test]
    fn test_generate_strategies_elementwise() {
        let optimizer = LoweringOptimizer::with_default_config();

        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![64.into()]);
        let result = -input;

        let strategies = optimizer.generate_strategies(&result);

        // デフォルト + ベクトル化候補 + 並列化候補 + アンロール候補
        // 最低でも2つ以上（デフォルト + 何らかの最適化）
        assert!(strategies.len() > 1);
    }

    #[test]
    fn test_can_vectorize_const_shape() {
        let optimizer = LoweringOptimizer::with_default_config();

        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![64.into()]);

        // 64は4, 8, 16で割り切れる
        assert!(optimizer.can_vectorize(&input, 0, 4));
        assert!(optimizer.can_vectorize(&input, 0, 8));
        assert!(optimizer.can_vectorize(&input, 0, 16));

        // 65は割り切れない
        let input2 = graph.input(DType::F32, vec![65.into()]);
        assert!(!optimizer.can_vectorize(&input2, 0, 4));
    }

    #[test]
    fn test_optimize_node_basic() {
        let optimizer = LoweringOptimizer::with_default_config();

        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![64.into()]);
        let result = -input;

        let (_optimized_node, _ast, cost) = optimizer.optimize_node(&result);

        // コストが正の値であることを確認
        assert!(cost > 0.0);
    }

    #[test]
    fn test_optimize_node_chooses_best() {
        let optimizer = LoweringOptimizer::with_default_config();

        let mut graph = Graph::new();
        let input1 = graph.input(DType::F32, vec![64.into()]);
        let input2 = graph.input(DType::F32, vec![64.into()]);
        let result = input1 + input2;

        let (_optimized_node, _ast, cost) = optimizer.optimize_node(&result);

        // 最適化が実行されたことを確認（コストが正の値）
        assert!(cost > 0.0);
    }
}
