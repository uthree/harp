//! Lowering戦略の最適化
//!
//! 複数の戦略候補を試して、ASTコストが最小になる戦略を選択します。

use crate::ast::helper::function;
use crate::ast::{AstNode, ConstLiteral, DType, Scope, VariableDecl};
use crate::graph::shape::Expr;
use crate::graph::{Graph, GraphNode, GraphOp, LoopStrategy};
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

    /// Graph全体を最適化してlowering
    ///
    /// すべての出力ノードに対して最適な戦略を探索し、
    /// 完全なASTプログラム（kernel_implとkernel_main）を生成します。
    ///
    /// # Arguments
    ///
    /// * `graph` - 変換するGraph
    ///
    /// # Returns
    ///
    /// 完全なASTプログラム
    pub fn optimize_and_lower(&self, graph: &Graph) -> AstNode {
        if self.config.enable_optimization {
            // 最適化あり：各出力ノードに対して戦略を探索
            self.lower_with_optimization(graph)
        } else {
            // 最適化なし：デフォルト戦略でlower
            self.lower_without_optimization(graph)
        }
    }

    /// 最適化ありでGraph全体をlower
    fn lower_with_optimization(&self, graph: &Graph) -> AstNode {
        let mut lowerer = RecursiveLowerer::new();

        // 入力ノードの変数名を事前マッピング
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(input_rc);
                lowerer.set_var_name(&input_node, format!("input_{}", i));
            }
        }

        // 出力ノードの変数名を事前マッピング
        for (i, output_node) in graph.outputs.iter().enumerate() {
            lowerer.set_var_name(output_node, format!("output_{}", i));
        }

        // 各出力ノードをlower
        // TODO: 将来的には各ノードに対して戦略を最適化する
        for output_node in &graph.outputs {
            lowerer.lower_node(output_node);
        }

        // プログラムを構築
        self.build_program(graph, lowerer)
    }

    /// 最適化なしでGraph全体をlower
    fn lower_without_optimization(&self, graph: &Graph) -> AstNode {
        let mut lowerer = RecursiveLowerer::new();

        // 入力ノードの変数名を事前マッピング
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(input_rc);
                lowerer.set_var_name(&input_node, format!("input_{}", i));
            }
        }

        // 出力ノードの変数名を事前マッピング
        for (i, output_node) in graph.outputs.iter().enumerate() {
            lowerer.set_var_name(output_node, format!("output_{}", i));
        }

        // 各出力ノードをlower
        for output_node in &graph.outputs {
            lowerer.lower_node(output_node);
        }

        // プログラムを構築
        self.build_program(graph, lowerer)
    }

    /// kernel_implとkernel_mainを含む完全なプログラムを構築
    fn build_program(&self, graph: &Graph, lowerer: RecursiveLowerer) -> AstNode {
        // kernel_impl関数を作成
        let kernel_impl = self.create_kernel_impl(graph, &lowerer);

        // kernel_main（エントリーポイント）関数を作成
        let kernel_main = self.create_kernel_main(graph);

        // Programノードを作成
        AstNode::program(vec![kernel_impl, kernel_main], "kernel_main")
    }

    /// kernel_impl関数を作成
    fn create_kernel_impl(&self, graph: &Graph, lowerer: &RecursiveLowerer) -> AstNode {
        // 引数リストを作成（入力 + 出力）
        let mut arguments = Vec::new();

        // 入力引数
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
                arguments.push((
                    format!("input_{}", i),
                    DType::Ptr(Box::new(input_rc.dtype.clone())),
                ));
            }
        }

        // 出力引数
        for (i, output_node) in graph.outputs.iter().enumerate() {
            arguments.push((
                format!("output_{}", i),
                DType::Ptr(Box::new(output_node.dtype.clone())),
            ));
        }

        // lowererから生成されたdeclarationsとstatementsを取得
        let declarations = lowerer.declarations.clone();
        let statements = lowerer.statements.clone();

        // kernel_impl関数を構築
        function(
            "kernel_impl".to_string(),
            arguments,
            DType::Void,
            Scope { declarations },
            statements,
        )
    }

    /// kernel_main関数（エントリーポイント）を作成
    fn create_kernel_main(&self, graph: &Graph) -> AstNode {
        let mut statements = Vec::new();
        let mut local_vars = Vec::new();

        let mut arg_index = 0;

        // 入力バッファの型キャスト
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
                let var_name = format!("input_{}", i);

                local_vars.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: DType::Ptr(Box::new(input_rc.dtype.clone())),
                    constant: false,
                    size_expr: None,
                });

                // cast: float* input_0 = (float*)bufs[0];
                statements.push(AstNode::Assign(
                    var_name,
                    Box::new(AstNode::Cast {
                        dtype: DType::Ptr(Box::new(input_rc.dtype.clone())),
                        expr: Box::new(AstNode::Load {
                            target: Box::new(AstNode::Var("bufs".to_string())),
                            index: Box::new(AstNode::Const(ConstLiteral::Usize(arg_index))),
                            vector_width: 1,
                        }),
                    }),
                ));
                arg_index += 1;
            }
        }

        // 出力バッファの型キャスト
        for (i, output_node) in graph.outputs.iter().enumerate() {
            let var_name = format!("output_{}", i);

            local_vars.push(VariableDecl {
                name: var_name.clone(),
                dtype: DType::Ptr(Box::new(output_node.dtype.clone())),
                constant: false,
                size_expr: None,
            });

            statements.push(AstNode::Assign(
                var_name.clone(),
                Box::new(AstNode::Cast {
                    dtype: DType::Ptr(Box::new(output_node.dtype.clone())),
                    expr: Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var("bufs".to_string())),
                        index: Box::new(AstNode::Const(ConstLiteral::Usize(arg_index))),
                        vector_width: 1,
                    }),
                }),
            ));
            arg_index += 1;
        }

        // kernel_implの呼び出し引数を作成
        let mut call_args = Vec::new();
        for (i, _) in graph.inputs.iter().enumerate() {
            call_args.push(AstNode::Var(format!("input_{}", i)));
        }
        for (i, _) in graph.outputs.iter().enumerate() {
            call_args.push(AstNode::Var(format!("output_{}", i)));
        }

        // kernel_implを呼び出し
        statements.push(AstNode::CallFunction {
            name: "kernel_impl".to_string(),
            args: call_args,
        });

        // エントリーポイント関数の引数
        let entry_args = vec![
            (
                "bufs".to_string(),
                DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
            ),
            ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize))),
        ];

        // kernel_main関数を構築
        function(
            "kernel_main".to_string(),
            entry_args,
            DType::Void,
            Scope {
                declarations: local_vars,
            },
            statements,
        )
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

    #[test]
    fn test_optimize_and_lower_simple_graph() {
        let optimizer = LoweringOptimizer::with_default_config();

        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);
        let result = -input;
        graph.output(result);

        let program = optimizer.optimize_and_lower(&graph);

        // Programノードが生成されたことを確認
        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_optimize_and_lower_with_optimization() {
        let config = LoweringConfig::default();
        let optimizer = LoweringOptimizer::new(config);

        let mut graph = Graph::new();
        let input1 = graph.input(DType::F32, vec![64.into()]);
        let input2 = graph.input(DType::F32, vec![64.into()]);
        let result = input1 + input2;
        graph.output(result);

        let program = optimizer.optimize_and_lower(&graph);

        // Programノードが生成されたことを確認
        if let AstNode::Program {
            entry_point,
            functions,
        } = program
        {
            assert_eq!(entry_point, "kernel_main");
            assert_eq!(functions.len(), 2); // kernel_impl + kernel_main
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_optimize_and_lower_without_optimization() {
        let config = LoweringConfig::no_optimization();
        let optimizer = LoweringOptimizer::new(config);

        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);
        let result = -input;
        graph.output(result);

        let program = optimizer.optimize_and_lower(&graph);

        // Programノードが生成されたことを確認
        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_optimize_and_lower_multiple_outputs() {
        let optimizer = LoweringOptimizer::with_default_config();

        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);
        let result1 = -input.clone();
        let result2 = input * GraphNode::f32(2.0);
        graph.output(result1);
        graph.output(result2);

        let program = optimizer.optimize_and_lower(&graph);

        // Programノードが生成されたことを確認
        if let AstNode::Program { functions, .. } = program {
            assert_eq!(functions.len(), 2); // kernel_impl + kernel_main

            // kernel_implが2つの出力を持つことを確認
            if let AstNode::Function { arguments, .. } = &functions[0] {
                // 1つの入力 + 2つの出力 = 3引数
                assert_eq!(arguments.len(), 3);
            } else {
                panic!("Expected Function node for kernel_impl");
            }
        } else {
            panic!("Expected Program node");
        }
    }
}
