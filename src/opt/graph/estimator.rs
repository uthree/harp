use crate::graph::{DType, ElementwiseStrategy, Graph, GraphNode, GraphOp};
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
            GraphOp::Input | GraphOp::Const(_) => 0.0,
            GraphOp::View(_) => 0.0, // View変更はゼロコスト
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
        match strategy {
            ElementwiseStrategy::Sequential { .. } => 1.0,
            ElementwiseStrategy::Thread { .. } => 0.3, // スレッド並列化で3倍高速化を想定
            ElementwiseStrategy::ThreadGroup { .. } => 0.1, // GPU並列化で10倍高速化を想定
        }
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
