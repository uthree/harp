use crate::graph::{ElementwiseOp, ElementwiseStrategy, Graph, GraphNode, GraphOp};
use crate::opt::ast::CostEstimator as AstCostEstimator;
use crate::opt::cost_utils::{log_sum_exp, log_sum_exp_iter};
use crate::opt::graph::GraphCostEstimator;
use std::collections::HashSet;

// メモリアクセスのコスト（L1キャッシュヒット想定、CPUサイクル）
const MEMORY_ACCESS_COST: f32 = 4.0;

// カーネル起動オーバーヘッド（CPUサイクル）
const KERNEL_LAUNCH_OVERHEAD: f32 = 100.0;

/// 簡単なコスト推定器（ノード数とメモリアクセスベース）
///
/// **重要**: このコスト推定器は対数スケール（log(CPUサイクル数)）でコストを返します。
/// 実際のサイクル数が必要な場合は、`result.exp()`を使用してください。
///
/// コストの単位はCPUサイクル数を想定しています。
/// AST評価関数と同じ単位系を使用することで、
/// グラフレベルとASTレベルの最適化を統一的に扱えます。
///
/// ノード数のペナルティ項を追加して、View変更の挿入による
/// 際限のないノード数の増加を防ぎます。
///
/// # ペナルティの計算
/// 対数スケールでは、ペナルティを次のように計算します：
/// ```text
/// final_cost = log_base_cost + penalty_coefficient * node_count
/// ```
/// これは元のスケールで `cost = base_cost * exp(penalty_coefficient * node_count)` に相当します。
pub struct SimpleCostEstimator {
    /// ノード数あたりのペナルティ係数（対数スケール、デフォルト: 0.01）
    /// 値が大きいほど、ノード数増加に対するペナルティが強くなります。
    node_count_penalty: f32,
}

impl SimpleCostEstimator {
    /// 新しいコスト推定器を作成
    pub fn new() -> Self {
        Self {
            node_count_penalty: 0.01, // デフォルト値
        }
    }

    /// ノード数ペナルティ係数を設定
    pub fn with_node_count_penalty(mut self, penalty: f32) -> Self {
        self.node_count_penalty = penalty;
        self
    }

    /// ElementwiseOp の演算コストを取得（log(CPUサイクル/演算)）
    fn elementwise_op_cost(&self, op: &ElementwiseOp) -> f32 {
        match op {
            ElementwiseOp::Add => 3.0_f32.ln(),
            ElementwiseOp::Mul => 4.0_f32.ln(),
            ElementwiseOp::Neg => 3.0_f32.ln(),
            ElementwiseOp::Max => 2.0_f32.ln(),
            ElementwiseOp::Rem => 25.0_f32.ln(),
            ElementwiseOp::Idiv => 25.0_f32.ln(),
            ElementwiseOp::Recip => 14.0_f32.ln(),
            ElementwiseOp::Sqrt => 15.0_f32.ln(),
            ElementwiseOp::Log2 => 40.0_f32.ln(),
            ElementwiseOp::Exp2 => 40.0_f32.ln(),
            ElementwiseOp::Sin => 50.0_f32.ln(),
        }
    }

    /// ReduceOp の演算コストを取得（log(CPUサイクル/演算)）
    fn reduce_op_cost(&self, op: &crate::graph::ReduceOp) -> f32 {
        match op {
            crate::graph::ReduceOp::Sum => 3.0_f32.ln(),
            crate::graph::ReduceOp::Max => 2.0_f32.ln(),
            crate::graph::ReduceOp::Prod => 4.0_f32.ln(),
        }
    }

    /// 各ノードのベースコストを取得（log(CPUサイクル)）
    fn node_base_cost(&self, node: &GraphNode) -> f32 {
        match &node.op {
            GraphOp::Input | GraphOp::Const(_) | GraphOp::ComplexConst { .. } => {
                // 入力/定数ノードは実行時コストなし（メモリは既に確保済み）
                f32::NEG_INFINITY // log(0)
            }
            GraphOp::View(_) => {
                // View変更は実行時コストゼロ（メタデータのみの変更）
                f32::NEG_INFINITY // log(0)
            }
            GraphOp::Contiguous { .. } => {
                // メモリコピーのコスト = 要素数 × (read + write) × MEMORY_ACCESS_COST
                // 対数スケール: log(num_elements * 2 * cost) = log(num_elements) + log(2 * cost)
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln()
            }
            GraphOp::Elementwise { op, .. } => {
                // 演算コスト = 要素数 × (演算コスト + メモリアクセスコスト)
                // 対数スケール: log(num_elements * (compute_cost + memory_cost))
                //             = log(num_elements) + log_sum_exp(log(compute_cost), log(memory_cost))
                let num_elements = self.compute_num_elements(node);
                let log_compute_cost = self.elementwise_op_cost(op);
                // 入力を読み、出力を書く
                let log_memory_cost = ((node.src.len() as f32 + 1.0) * MEMORY_ACCESS_COST).ln();
                num_elements.ln() + log_sum_exp(log_compute_cost, log_memory_cost)
            }
            GraphOp::Reduce { op, .. } => {
                // Reduceは入力サイズに依存
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                let log_reduce_cost = self.reduce_op_cost(op);
                // 入力読み取り + 縮約演算
                // log(num_elements * (MEMORY_ACCESS_COST + reduce_cost))
                num_elements.ln() + log_sum_exp(MEMORY_ACCESS_COST.ln(), log_reduce_cost)
            }
            GraphOp::Cumulative { .. } => {
                // Cumulativeは逐次依存性が高い（並列化が困難）
                // 累積和を想定（Sumのコスト）
                let num_elements = self.compute_num_elements(node);
                let log_cumulative_cost = 3.0_f32.ln(); // Sumのコスト
                // 各要素で読み取り + 演算 + 書き込み
                // log(num_elements * (2 * MEMORY_ACCESS_COST + cumulative_cost))
                num_elements.ln()
                    + log_sum_exp((2.0 * MEMORY_ACCESS_COST).ln(), log_cumulative_cost)
            }
            GraphOp::FusedElementwise { expr, .. } => {
                // 融合演算は中間バッファを節約
                let num_elements = self.compute_num_elements(node);
                let log_ops_cost = self.ast_expr_cost(expr);
                // 融合により中間バッファへのメモリアクセスが削減される
                // 入力読み取り + 演算 + 出力書き込みのみ
                let log_memory_cost = ((node.src.len() as f32 + 1.0) * MEMORY_ACCESS_COST).ln();
                num_elements.ln() + log_sum_exp(log_ops_cost, log_memory_cost)
            }
            GraphOp::FusedElementwiseReduce {
                expr, reduce_op, ..
            } => {
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                let log_elementwise_cost = self.ast_expr_cost(expr);
                let log_reduce_cost = self.reduce_op_cost(reduce_op);
                // 入力読み取り + elementwise演算 + reduce演算
                num_elements.ln()
                    + log_sum_exp_iter(vec![
                        MEMORY_ACCESS_COST.ln(),
                        log_elementwise_cost,
                        log_reduce_cost,
                    ])
            }
            GraphOp::FusedElementwiseCumulative { expr, .. } => {
                // FusedElementwiseCumulativeはCumulativeと同様の逐次依存性
                // + elementwise演算のコスト
                let num_elements = self.compute_num_elements(node);
                let log_elementwise_cost = self.ast_expr_cost(expr);
                let log_cumulative_cost = 3.0_f32.ln(); // Sumのコスト
                // 各要素で読み取り + elementwise演算 + 累積演算 + 書き込み
                num_elements.ln()
                    + log_sum_exp_iter(vec![
                        (2.0 * MEMORY_ACCESS_COST).ln(),
                        log_elementwise_cost,
                        log_cumulative_cost,
                    ])
            }
            GraphOp::FusedReduce { ops, .. } => {
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                let log_reduce_cost =
                    log_sum_exp_iter(ops.iter().map(|op| self.reduce_op_cost(op)));
                // 複数のreduce演算を融合
                num_elements.ln() + log_sum_exp(MEMORY_ACCESS_COST.ln(), log_reduce_cost)
            }
            GraphOp::Pad { .. } => {
                // Padは出力バッファの初期化 + 入力データのコピー
                // コスト = 出力要素数 × (初期化 + コピー) × MEMORY_ACCESS_COST
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln()
            }
            GraphOp::Slice { .. } => {
                // Sliceは入力からのコピーのみ（出力要素数ベース）
                // コスト = 出力要素数 × (read + write) × MEMORY_ACCESS_COST
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln()
            }
            GraphOp::Concat { .. } => {
                // Concatは全入力からのコピー（出力要素数ベース）
                // コスト = 出力要素数 × (read + write) × MEMORY_ACCESS_COST
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln()
            }
            GraphOp::Fold { .. } => {
                // Fold: col2im、重複部分の加算が必要
                // unfold演算の逆操作なので、高コスト
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (3.0 * MEMORY_ACCESS_COST).ln() // 読み込み + 書き込み + 加算
            }
            GraphOp::Rand { .. } => {
                // 乱数初期化: 各要素に乱数生成 + 書き込み
                // 乱数生成のコストは比較的高い
                let num_elements = self.compute_num_elements(node);
                let log_rand_cost = 10.0_f32.ln(); // 乱数生成は比較的高コスト
                num_elements.ln() + log_sum_exp(log_rand_cost, MEMORY_ACCESS_COST.ln())
            }
            GraphOp::Arange { .. } => {
                // 連番初期化: 各要素にインデックス値を書き込み
                // 非常に軽量（書き込みのみ）
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + MEMORY_ACCESS_COST.ln()
            }
            GraphOp::Cast { .. } => {
                // 型変換: 各要素をキャスト
                // 非常に軽量（読み込み + キャスト + 書き込み）
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln()
            }
            GraphOp::Real { .. } | GraphOp::Imag { .. } => {
                // 複素数から実部/虚部を抽出
                // 読み込み + 書き込み（stride 2でのアクセス）
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln()
            }
            GraphOp::ComplexFromParts { .. } => {
                // 実部と虚部から複素数を構築
                // 2つの入力を読み込み + インターリーブして書き込み
                let num_elements = self.compute_num_elements(node);
                num_elements.ln() + (3.0 * MEMORY_ACCESS_COST).ln()
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

    /// AstNode式の演算コストを推定（対数スケール）
    #[allow(clippy::only_used_in_recursion)]
    fn ast_expr_cost(&self, expr: &crate::ast::AstNode) -> f32 {
        use crate::ast::AstNode;
        match expr {
            AstNode::Wildcard(_) | AstNode::Const(_) | AstNode::Var(_) => {
                // リーフノードはコストなし
                f32::NEG_INFINITY
            }
            AstNode::Add(left, right)
            | AstNode::Mul(left, right)
            | AstNode::Max(left, right)
            | AstNode::Rem(left, right)
            | AstNode::Idiv(left, right) => {
                // 基本演算コスト + 子ノードのコスト
                let op_cost = 3.0_f32.ln(); // 基本演算コスト
                let left_cost = self.ast_expr_cost(left);
                let right_cost = self.ast_expr_cost(right);
                log_sum_exp_iter(vec![op_cost, left_cost, right_cost])
            }
            AstNode::Recip(operand) | AstNode::Sqrt(operand) => {
                // 重い演算 + 子ノードのコスト
                let op_cost = 20.0_f32.ln();
                let child_cost = self.ast_expr_cost(operand);
                log_sum_exp(op_cost, child_cost)
            }
            AstNode::Log2(operand) | AstNode::Exp2(operand) | AstNode::Sin(operand) => {
                // 非常に重い演算
                let op_cost = 50.0_f32.ln();
                let child_cost = self.ast_expr_cost(operand);
                log_sum_exp(op_cost, child_cost)
            }
            AstNode::Cast(operand, _) => {
                // キャストはほぼ無料
                self.ast_expr_cost(operand)
            }
            _ => {
                // その他の複雑なノードは基本コスト
                3.0_f32.ln()
            }
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
        let node_count = nodes.len();
        let mut log_costs = Vec::new();

        for node in &nodes {
            let log_base_cost = self.node_base_cost(node);

            // 並列化戦略によるコスト削減を適用
            let log_strategy_factor = if !node.elementwise_strategies.is_empty() {
                // 各軸の戦略の平均を取る
                let sum: f32 = node
                    .elementwise_strategies
                    .iter()
                    .map(|s| self.strategy_cost_factor(s))
                    .sum();
                let avg = sum / node.elementwise_strategies.len() as f32;
                avg.ln()
            } else {
                0.0 // log(1) = 0
            };

            // 対数スケールでの乗算: log(base_cost * strategy_factor) = log_base_cost + log_strategy_factor
            log_costs.push(log_base_cost + log_strategy_factor);
        }

        // カーネル起動オーバーヘッド（出力ノード数に比例）
        let num_outputs = graph.outputs().len() as f32;
        let log_kernel_overhead = num_outputs.ln() + KERNEL_LAUNCH_OVERHEAD.ln();

        // すべてのコストを合計
        log_costs.push(log_kernel_overhead);
        let log_base_cost = log_sum_exp_iter(log_costs);

        // ノード数のペナルティを対数スケールで直接加算
        // final_cost = log_base_cost + penalty_coefficient * node_count
        // これは元のスケールで cost = base_cost * exp(penalty_coefficient * node_count)
        let penalty = self.node_count_penalty * node_count as f32;

        log_base_cost + penalty
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

    #[test]
    fn test_simple_node_count_penalty() {
        let estimator = SimpleCostEstimator::new().with_node_count_penalty(1.0);

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

        let log_cost1 = estimator.estimate(&graph1);
        let log_cost2 = estimator.estimate(&graph2);

        // ノード数が多い方がコストが高いはず（ペナルティが効いている）
        assert!(log_cost2 > log_cost1);
    }

    #[test]
    fn test_simple_zero_penalty() {
        let estimator_no_penalty = SimpleCostEstimator::new().with_node_count_penalty(0.0);
        let estimator_with_penalty = SimpleCostEstimator::new().with_node_count_penalty(10.0);

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![100])
            .build();
        graph.output("c", a + b);

        let cost_no_penalty = estimator_no_penalty.estimate(&graph);
        let cost_with_penalty = estimator_with_penalty.estimate(&graph);

        // ペナルティありの場合、コストが上昇するはず
        assert!(cost_with_penalty > cost_no_penalty);
    }
}

/// ASTベースのコスト推定器
///
/// **重要**: このコスト推定器は対数スケール（log(CPUサイクル数)）でコストを返します。
///
/// グラフ全体をProgramに変換してからコストを推定します。
/// より正確なコスト推定が可能ですが、SimpleCostEstimatorよりも計算コストが高いです。
///
/// ノード数のペナルティ項を追加して、View変更の挿入による
/// 際限のないノード数の増加を防ぎます。
///
/// # ペナルティの計算
/// 対数スケールでは、ペナルティを次のように計算します：
/// ```text
/// final_cost = log_ast_cost + penalty_coefficient * node_count
/// ```
/// これは元のスケールで `cost = ast_cost * exp(penalty_coefficient * node_count)` に相当します。
/// penalty_coefficientが小さい場合（例：0.001）、近似的に
/// `cost ≈ ast_cost * (1 + penalty_coefficient * node_count)` となります。
pub struct AstBasedCostEstimator<E: AstCostEstimator> {
    ast_estimator: E,
    /// ノード数あたりのペナルティ係数（対数スケール、デフォルト: 0.01）
    /// 値が大きいほど、ノード数増加に対するペナルティが強くなります。
    node_count_penalty: f32,
}

impl<E: AstCostEstimator> AstBasedCostEstimator<E> {
    /// 新しいASTベースコスト推定器を作成
    pub fn new(ast_estimator: E) -> Self {
        Self {
            ast_estimator,
            node_count_penalty: 0.05, // 対数スケールでの適切な値
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
        let log_ast_cost = self.ast_estimator.estimate(&program);

        // ノード数のペナルティを対数スケールで直接加算
        // final_cost = log_ast_cost + penalty_coefficient * node_count
        // これは元のスケールで cost = ast_cost * exp(penalty_coefficient * node_count)
        // penalty_coefficientが小さい場合、近似的に cost ≈ ast_cost * (1 + penalty_coefficient * node_count)
        let node_count = self.count_nodes(graph);
        let penalty = self.node_count_penalty * node_count as f32;

        log_ast_cost + penalty
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

        let log_cost1 = estimator.estimate(&graph1);
        let log_cost2 = estimator.estimate(&graph2);

        // グラフ全体をProgramに変換する際、shapeが定数の場合はループ回数も定数になる
        // したがって、要素数が異なる場合は総コストも異なるが、要素あたりのコストは同じになる
        // 対数スケール: log(per_element_cost) = log(total_cost) - log(num_elements)
        let log_per_element_cost1 = log_cost1 - 10.0_f32.ln();
        let log_per_element_cost2 = log_cost2 - 1000.0_f32.ln();

        // 要素あたりのコストがほぼ同じであることを確認（対数スケール）
        // 注: 関数定義オーバーヘッドは一定なので、要素数が少ない方が要素あたりのコストが高くなる
        // そのため、閾値を緩めに設定（0.5以下）
        let diff = (log_per_element_cost1 - log_per_element_cost2).abs();
        assert!(
            diff < 0.5,
            "Per-element costs should be similar (log scale): log_cost1={} (log {}/elem), log_cost2={} (log {}/elem), diff={}",
            log_cost1,
            log_per_element_cost1,
            log_cost2,
            log_per_element_cost2,
            diff
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

        let log_cost1 = estimator.estimate(&graph1);
        let log_cost2 = estimator.estimate(&graph2);

        // ノード数が多い方がコストが高いはず（ペナルティが効いている）
        assert!(log_cost2 > log_cost1);

        // 対数スケールでは、コストの差は比率の対数
        // log(cost2) - log(cost1) = log(cost2/cost1)
        // ノード数の差は2なので、元のスケールでペナルティ差は約2.0
        // 対数スケールでは log(2) ≈ 0.693
        let log_cost_diff = log_cost2 - log_cost1;
        // AST costの差もあるので、少なくとも log(2) 程度の差があるはず
        assert!(log_cost_diff >= 0.6, "log_cost_diff={}", log_cost_diff);
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
        // ペナルティを大きくして効果を明確にする
        let estimator_with_penalty =
            AstBasedCostEstimator::new(AstSimpleCostEstimator::new()).with_node_count_penalty(10.0);

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

        let log_cost1_no_penalty = estimator_no_penalty.estimate(&graph1);
        let log_cost2_no_penalty = estimator_no_penalty.estimate(&graph2);
        let log_cost1_with_penalty = estimator_with_penalty.estimate(&graph1);
        let log_cost2_with_penalty = estimator_with_penalty.estimate(&graph2);

        // ペナルティなしの場合、AST costのみの差
        // graph2の方が複雑なので、コストは高い（対数スケール）
        assert!(log_cost2_no_penalty > log_cost1_no_penalty);

        // ペナルティありの場合、どちらのグラフもコストが上昇するはず
        assert!(log_cost1_with_penalty > log_cost1_no_penalty);
        assert!(log_cost2_with_penalty > log_cost2_no_penalty);

        // ペナルティがゼロの場合とゼロでない場合で、コストが異なることを確認
        // （対数スケールでは、ペナルティの相対的な影響は元のコストに依存する）
        // graph2の方がノード数が多いので、絶対的なペナルティは大きいが、
        // ast_costも大きいため、相対的な影響（対数での差）は必ずしも大きくない
        let cost1_increase = log_cost1_with_penalty - log_cost1_no_penalty;
        let cost2_increase = log_cost2_with_penalty - log_cost2_no_penalty;
        // どちらも正の値（ペナルティの効果がある）
        assert!(cost1_increase > 0.0);
        assert!(cost2_increase > 0.0);
    }
}
