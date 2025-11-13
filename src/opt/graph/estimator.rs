use crate::graph::{DType, ElementwiseStrategy, Graph, GraphNode, GraphOp};
use crate::opt::ast::CostEstimator as AstCostEstimator;
use crate::opt::graph::GraphCostEstimator;
use std::collections::HashSet;

/// 簡単なコスト推定器（ノード数とメモリアクセスベース）
pub struct SimpleCostEstimator;

impl SimpleCostEstimator {
    /// 新しいコスト推定器を作成
    pub fn new() -> Self {
        Self
    }

    /// 各ノードのベースコストを取得
    fn node_base_cost(&self, node: &GraphNode) -> f32 {
        match &node.op {
            GraphOp::Input | GraphOp::Const(_) => 0.1,
            GraphOp::View(_) => 0.1, // View変更はほぼゼロ
            GraphOp::Contiguous { .. } => {
                // メモリコピーのコスト = 要素数 × dtype size × 2 (read + write)
                let num_elements = self.compute_num_elements(node);
                let dtype_size = self.dtype_size(&node.dtype);
                num_elements * dtype_size * 2.0
            }
            GraphOp::Elementwise { .. } => {
                // 演算コスト = 要素数 × 演算コスト
                let num_elements = self.compute_num_elements(node);
                let compute_cost = 1.0; // 基本演算コスト
                num_elements * compute_cost
            }
            GraphOp::Reduce { .. } => {
                // Reduceは入力サイズに依存
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                num_elements * 1.5 // 縮約は若干重い
            }
            GraphOp::Cumulative { .. } => {
                // Cumulativeは逐次依存性が高い
                let num_elements = self.compute_num_elements(node);
                num_elements * 2.0
            }
            GraphOp::FusedElementwise { ops, .. } => {
                // 融合演算は中間バッファを節約
                let num_elements = self.compute_num_elements(node);
                let num_ops = ops.len() as f32;
                num_elements * num_ops * 0.8 // 融合により20%削減
            }
            GraphOp::FusedElementwiseReduce {
                elementwise_ops, ..
            } => {
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                let num_ops = elementwise_ops.len() as f32;
                num_elements * (num_ops + 1.5) * 0.8
            }
            GraphOp::FusedReduce { ops, .. } => {
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                let num_ops = ops.len() as f32;
                num_elements * num_ops * 1.5 * 0.9 // 融合により10%削減
            }
        }
    }

    /// 並列化戦略によるコスト係数を取得
    fn strategy_cost_factor(&self, strategy: &ElementwiseStrategy) -> f32 {
        // ベースとなる並列化レベルの係数
        let base_factor = match strategy {
            ElementwiseStrategy::Sequential { .. } => 1.0,
            ElementwiseStrategy::Thread { .. } => 0.3, // スレッド並列化で3倍高速化を想定
            ElementwiseStrategy::ThreadGroup { .. } => 0.1, // GPU並列化で10倍高速化を想定
        };

        // SIMD幅とアンローリング係数を取得
        let (simd_width, unroll_factor) = match strategy {
            ElementwiseStrategy::Sequential {
                simd_width,
                unroll_factor,
            }
            | ElementwiseStrategy::Thread {
                simd_width,
                unroll_factor,
            }
            | ElementwiseStrategy::ThreadGroup {
                simd_width,
                unroll_factor,
            } => (*simd_width, *unroll_factor),
        };

        // SIMD効果: simd_width倍の並列化（ただし効率は85%と仮定）
        let simd_factor = if simd_width > 1 {
            1.0 / (simd_width as f32 * 0.85)
        } else {
            1.0
        };

        // アンローリング効果: ループオーバーヘッド削減
        // unroll_factor倍にループを展開すると、ループ制御コストが1/unroll_factor
        // ただし効果は小さいので、5%程度の改善と仮定
        let unroll_effect = if unroll_factor > 1 {
            1.0 - (0.05 * (unroll_factor as f32).log2())
        } else {
            1.0
        };

        base_factor * simd_factor * unroll_effect
    }

    /// ノードの要素数を計算
    fn compute_num_elements(&self, node: &GraphNode) -> f32 {
        use crate::graph::shape::Expr;

        let shape = node.view.shape();
        let mut num_elements = 1.0;
        for dim in shape {
            // Exprを評価してusizeに変換（簡易実装）
            match dim {
                Expr::Const(size) => {
                    num_elements *= *size as f32;
                }
                _ => {
                    // 評価できない場合はデフォルト値
                    num_elements *= 100.0;
                }
            }
        }
        num_elements
    }

    /// DTypeのサイズを取得（バイト）
    fn dtype_size(&self, dtype: &DType) -> f32 {
        match dtype {
            DType::F32 => 4.0,
            DType::Unknown => 4.0, // デフォルトで4バイトと仮定
        }
    }

    /// グラフ内の全ノードを収集（トポロジカル順）
    fn collect_all_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const crate::graph::GraphNodeData>,
            nodes: &mut Vec<GraphNode>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            // 先に依存ノードを訪問
            for src in &node.src {
                visit(src, visited, nodes);
            }

            nodes.push(node.clone());
        }

        for output in graph.outputs().values() {
            visit(output, &mut visited, &mut nodes);
        }

        nodes
    }
}

impl Default for SimpleCostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCostEstimator for SimpleCostEstimator {
    fn estimate(&self, graph: &Graph) -> f32 {
        let nodes = self.collect_all_nodes(graph);
        let mut total_cost = 0.0;

        for node in &nodes {
            let base_cost = self.node_base_cost(node);

            // 並列化戦略によるコスト削減を適用
            let strategy_factor = if !node.elementwise_strategies.is_empty() {
                // 各軸の戦略の平均を取る
                let sum: f32 = node
                    .elementwise_strategies
                    .iter()
                    .map(|s| self.strategy_cost_factor(s))
                    .sum();
                sum / node.elementwise_strategies.len() as f32
            } else {
                1.0
            };

            total_cost += base_cost * strategy_factor;
        }

        // カーネル起動オーバーヘッド（出力ノード数に比例）
        let kernel_overhead = graph.outputs().len() as f32 * 10.0;
        total_cost + kernel_overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_simple_cost_estimator() {
        let estimator = SimpleCostEstimator::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let c = a + b;
        graph.output("c", c);

        let cost = estimator.estimate(&graph);
        // コストは正の値であるべき
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_comparison() {
        let estimator = SimpleCostEstimator::new();

        // 小さいグラフ
        let mut graph1 = Graph::new();
        let a1 = graph1
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b1 = graph1
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        graph1.output("c", a1 + b1);

        // 大きいグラフ
        let mut graph2 = Graph::new();
        let a2 = graph2
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![1000])
            .build();
        let b2 = graph2
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![1000])
            .build();
        graph2.output("c", a2 + b2);

        let cost1 = estimator.estimate(&graph1);
        let cost2 = estimator.estimate(&graph2);

        // 大きいグラフの方がコストが高いはず
        assert!(cost2 > cost1);
    }
}

/// ASTベースのコスト推定器
///
/// グラフ全体をProgramに変換してからコストを推定します。
/// より正確なコスト推定が可能ですが、SimpleCostEstimatorよりも計算コストが高いです。
///
/// ノード数のペナルティ項を追加して、View変更の挿入による
/// 際限のないノード数の増加を防ぎます。
pub struct AstBasedCostEstimator<E: AstCostEstimator> {
    ast_estimator: E,
    /// ノード数あたりのペナルティ係数（デフォルト: 0.0001）
    node_count_penalty: f32,
}

impl<E: AstCostEstimator> AstBasedCostEstimator<E> {
    /// 新しいASTベースコスト推定器を作成
    pub fn new(ast_estimator: E) -> Self {
        Self {
            ast_estimator,
            node_count_penalty: 0.0001,
        }
    }

    /// ノード数ペナルティ係数を設定
    pub fn with_node_count_penalty(mut self, penalty: f32) -> Self {
        self.node_count_penalty = penalty;
        self
    }

    /// グラフ内の全ノード数を数える
    fn count_nodes(&self, graph: &Graph) -> usize {
        let mut visited = HashSet::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const crate::graph::GraphNodeData>,
        ) -> usize {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return 0;
            }
            visited.insert(ptr);

            let mut count = 1;
            for src in &node.src {
                count += visit(src, visited);
            }
            count
        }

        let mut total = 0;
        for output in graph.outputs().values() {
            total += visit(output, &mut visited);
        }
        total
    }
}

impl<E: AstCostEstimator> GraphCostEstimator for AstBasedCostEstimator<E> {
    fn estimate(&self, graph: &Graph) -> f32 {
        // グラフ全体をProgramに変換
        let program = crate::lowerer::lower(graph.clone());
        let ast_cost = self.ast_estimator.estimate(&program);

        // ノード数のペナルティを追加
        let node_count = self.count_nodes(graph);
        let node_penalty = node_count as f32 * self.node_count_penalty;

        ast_cost + node_penalty
    }
}

#[cfg(test)]
mod ast_based_tests {
    use super::*;
    use crate::graph::{DType, Graph};
    use crate::opt::ast::SimpleCostEstimator as AstSimpleCostEstimator;

    #[test]
    fn test_ast_based_cost_estimator() {
        let ast_estimator = AstSimpleCostEstimator::new();
        let estimator = AstBasedCostEstimator::new(ast_estimator);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let c = a + b;
        graph.output("c", c);

        let cost = estimator.estimate(&graph);
        // コストは正の値であるべき
        assert!(cost > 0.0);
    }

    #[test]
    fn test_ast_cost_same_structure() {
        let ast_estimator = AstSimpleCostEstimator::new();
        let estimator = AstBasedCostEstimator::new(ast_estimator);

        // 小さいグラフ（10要素）
        let mut graph1 = Graph::new();
        let a1 = graph1
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b1 = graph1
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        graph1.output("c", a1 + b1);

        // 大きいグラフ（1000要素）- 生成されるASTは同じ構造
        let mut graph2 = Graph::new();
        let a2 = graph2
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![1000])
            .build();
        let b2 = graph2
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![1000])
            .build();
        graph2.output("c", a2 + b2);

        let cost1 = estimator.estimate(&graph1);
        let cost2 = estimator.estimate(&graph2);

        // グラフ全体をProgramに変換するため、shapeはパラメータになり
        // 生成されるASTは同じ構造になる（ループ回数は変数なので100回と推定）
        // したがってコストもほぼ同じになる（浮動小数点の誤差を考慮）
        assert!(
            (cost1 - cost2).abs() < 1e-3,
            "Costs should be similar: cost1={}, cost2={}",
            cost1,
            cost2
        );
    }

    #[test]
    fn test_ast_cost_multiple_ops() {
        let ast_estimator = AstSimpleCostEstimator::new();
        let estimator = AstBasedCostEstimator::new(ast_estimator);

        // 単一演算
        let mut graph1 = Graph::new();
        let a1 = graph1
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let b1 = graph1
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        graph1.output("c", a1 + b1);

        // 複数演算（a + b） * c
        let mut graph2 = Graph::new();
        let a2 = graph2
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let b2 = graph2
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let c2 = graph2
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let add = a2 + b2;
        let mul = add * c2;
        graph2.output("out", mul);

        let cost1 = estimator.estimate(&graph1);
        let cost2 = estimator.estimate(&graph2);

        // 複数演算の方がコストが高いはず
        assert!(cost2 > cost1);
    }

    #[test]
    fn test_node_count_penalty() {
        let ast_estimator = AstSimpleCostEstimator::new();
        let estimator = AstBasedCostEstimator::new(ast_estimator).with_node_count_penalty(1.0);

        // 少ないノード (3ノード: a, b, c)
        let mut graph1 = Graph::new();
        let a1 = graph1
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let b1 = graph1
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        graph1.output("c", a1 + b1);

        // 多いノード (5ノード: a, b, c, d, e)
        let mut graph2 = Graph::new();
        let a2 = graph2
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let b2 = graph2
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let c2 = graph2
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let d2 = a2 + b2;
        let e2 = d2 * c2;
        graph2.output("out", e2);

        let cost1 = estimator.estimate(&graph1);
        let cost2 = estimator.estimate(&graph2);

        // ノード数が多い方がコストが高いはず（ペナルティが効いている）
        assert!(cost2 > cost1);

        // ノード数の差は2なので、ペナルティ差は約2.0のはず
        let cost_diff = cost2 - cost1;
        // AST costの差もあるので、少なくとも2.0以上の差があるはず
        assert!(cost_diff >= 2.0);
    }

    #[test]
    fn test_node_count_penalty_with_views() {
        let ast_estimator = AstSimpleCostEstimator::new();
        let estimator = AstBasedCostEstimator::new(ast_estimator).with_node_count_penalty(0.5);

        // Viewノードなし
        let mut graph1 = Graph::new();
        let a1 = graph1
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();
        let b1 = graph1
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();
        graph1.output("c", a1 + b1);

        // Viewノードあり（転置を追加）
        let mut graph2 = Graph::new();
        let a2 = graph2
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();
        let b2 = graph2
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // a, bを転置
        let a2_t = a2.view(a2.view.clone().permute(vec![1, 0]));
        let b2_t = b2.view(b2.view.clone().permute(vec![1, 0]));

        // 転置されたテンソルで加算
        let c2_t = a2_t + b2_t;

        // 結果を逆転置
        let c2 = c2_t.view(c2_t.view.clone().permute(vec![1, 0]));

        graph2.output("c", c2);

        let cost1 = estimator.estimate(&graph1);
        let cost2 = estimator.estimate(&graph2);

        // Viewノードが追加された分、コストが高くなるはず
        assert!(cost2 > cost1);
    }

    #[test]
    fn test_zero_penalty() {
        let estimator_no_penalty =
            AstBasedCostEstimator::new(AstSimpleCostEstimator::new()).with_node_count_penalty(0.0);
        let estimator_with_penalty =
            AstBasedCostEstimator::new(AstSimpleCostEstimator::new()).with_node_count_penalty(1.0);

        let mut graph1 = Graph::new();
        let a1 = graph1
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let b1 = graph1
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        graph1.output("c", a1 + b1);

        let mut graph2 = Graph::new();
        let a2 = graph2
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let b2 = graph2
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let c2 = graph2
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let d2 = a2 + b2;
        let e2 = d2 * c2;
        graph2.output("out", e2);

        let cost1_no_penalty = estimator_no_penalty.estimate(&graph1);
        let cost2_no_penalty = estimator_no_penalty.estimate(&graph2);
        let cost1_with_penalty = estimator_with_penalty.estimate(&graph1);
        let cost2_with_penalty = estimator_with_penalty.estimate(&graph2);

        // ペナルティなしの場合、AST costのみの差
        // graph2の方が複雑なので、コストは高い
        assert!(cost2_no_penalty > cost1_no_penalty);

        // ペナルティありの場合、ノード数の差も加わるので、さらに差が大きくなる
        let diff_no_penalty = cost2_no_penalty - cost1_no_penalty;
        let diff_with_penalty = cost2_with_penalty - cost1_with_penalty;
        assert!(diff_with_penalty > diff_no_penalty);
    }
}
