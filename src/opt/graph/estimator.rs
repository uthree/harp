use crate::ast::AstNode;
use crate::graph::{ElementwiseOp, Graph, GraphNode, GraphOp};
use crate::opt::ast::CostEstimator as AstCostEstimator;
use crate::opt::ast::SimpleCostEstimator as AstSimpleCostEstimator;
use crate::opt::cost_utils::{log_sum_exp, log_sum_exp_iter};
use crate::opt::graph::GraphCostEstimator;
use std::collections::HashSet;

// メモリアクセスのコスト（L1キャッシュヒット想定、CPUサイクル）
const MEMORY_ACCESS_COST: f32 = 4.0;

// カーネル起動オーバーヘッド（CPUサイクル）
// 複数カーネルよりも1つのProgram/Functionを優先するため、高めに設定
const KERNEL_LAUNCH_OVERHEAD: f32 = 1000.0;

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
            // ノード数ペナルティを高く設定して、複数ノードより
            // 1つのKernelノードへの融合を優先させる
            node_count_penalty: 0.5,
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
            GraphOp::Buffer { .. } | GraphOp::Const(_) | GraphOp::ComplexConst { .. } => {
                // 入力/定数ノードは実行時コストなし（メモリは既に確保済み）
                f32::NEG_INFINITY // log(0)
            }
            GraphOp::View(_) => {
                // View変更は実行時コストゼロ（メタデータのみの変更）
                f32::NEG_INFINITY // log(0)
            }
            GraphOp::Contiguous => {
                // メモリコピーのコスト = 要素数 × (read + write) × MEMORY_ACCESS_COST
                // 対数スケール: log(num_elements * 2 * cost) = log(num_elements) + log(2 * cost)
                let num_elements = self.compute_num_elements(node);
                // ContiguousもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln() + lowering_penalty
            }
            GraphOp::Elementwise { op, .. } => {
                // 演算コスト = 要素数 × (演算コスト + メモリアクセスコスト)
                // 対数スケール: log(num_elements * (compute_cost + memory_cost))
                //             = log(num_elements) + log_sum_exp(log(compute_cost), log(memory_cost))
                let num_elements = self.compute_num_elements(node);
                let log_compute_cost = self.elementwise_op_cost(op);
                // 入力を読み、出力を書く
                let log_memory_cost = ((node.src.len() as f32 + 1.0) * MEMORY_ACCESS_COST).ln();
                // ElementwiseはKernelにloweringされるべきなので、大きなペナルティを追加
                // これによりオプティマイザはElementwiseをKernelに変換する方向に進む
                let elementwise_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln()
                    + log_sum_exp(log_compute_cost, log_memory_cost)
                    + elementwise_penalty
            }
            GraphOp::Reduce { op, .. } => {
                // Reduceは入力サイズに依存
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                let log_reduce_cost = self.reduce_op_cost(op);
                // 入力読み取り + 縮約演算
                // log(num_elements * (MEMORY_ACCESS_COST + reduce_cost))
                // ReduceもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln()
                    + log_sum_exp(MEMORY_ACCESS_COST.ln(), log_reduce_cost)
                    + lowering_penalty
            }
            GraphOp::Cumulative { .. } => {
                // Cumulativeは逐次依存性が高い（並列化が困難）
                // 累積和を想定（Sumのコスト）
                let num_elements = self.compute_num_elements(node);
                let log_cumulative_cost = 3.0_f32.ln(); // Sumのコスト
                // 各要素で読み取り + 演算 + 書き込み
                // log(num_elements * (2 * MEMORY_ACCESS_COST + cumulative_cost))
                // CumulativeもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln()
                    + log_sum_exp((2.0 * MEMORY_ACCESS_COST).ln(), log_cumulative_cost)
                    + lowering_penalty
            }
            GraphOp::FusedElementwise { expr, .. } => {
                // 融合演算は中間バッファを節約
                let num_elements = self.compute_num_elements(node);
                let log_ops_cost = self.ast_expr_cost(expr);
                // 融合により中間バッファへのメモリアクセスが削減される
                // 入力読み取り + 演算 + 出力書き込みのみ
                let log_memory_cost = ((node.src.len() as f32 + 1.0) * MEMORY_ACCESS_COST).ln();
                // FusedElementwiseもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln() + log_sum_exp(log_ops_cost, log_memory_cost) + lowering_penalty
            }
            GraphOp::FusedElementwiseReduce {
                expr, reduce_op, ..
            } => {
                let input = &node.src[0];
                let num_elements = self.compute_num_elements(input);
                let log_elementwise_cost = self.ast_expr_cost(expr);
                let log_reduce_cost = self.reduce_op_cost(reduce_op);
                // 入力読み取り + elementwise演算 + reduce演算
                // FusedElementwiseReduceもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln()
                    + log_sum_exp_iter(vec![
                        MEMORY_ACCESS_COST.ln(),
                        log_elementwise_cost,
                        log_reduce_cost,
                    ])
                    + lowering_penalty
            }
            GraphOp::FusedElementwiseCumulative { expr, .. } => {
                // FusedElementwiseCumulativeはCumulativeと同様の逐次依存性
                // + elementwise演算のコスト
                let num_elements = self.compute_num_elements(node);
                let log_elementwise_cost = self.ast_expr_cost(expr);
                let log_cumulative_cost = 3.0_f32.ln(); // Sumのコスト
                // 各要素で読み取り + elementwise演算 + 累積演算 + 書き込み
                // FusedElementwiseCumulativeもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln()
                    + log_sum_exp_iter(vec![
                        (2.0 * MEMORY_ACCESS_COST).ln(),
                        log_elementwise_cost,
                        log_cumulative_cost,
                    ])
                    + lowering_penalty
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
                // SliceもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln() + lowering_penalty
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
            GraphOp::Rand => {
                // 乱数初期化: 各要素に乱数生成 + 書き込み
                // 乱数生成のコストは比較的高い
                let num_elements = self.compute_num_elements(node);
                let log_rand_cost = 10.0_f32.ln(); // 乱数生成は比較的高コスト
                // RandもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln()
                    + log_sum_exp(log_rand_cost, MEMORY_ACCESS_COST.ln())
                    + lowering_penalty
            }
            GraphOp::Arange => {
                // 連番初期化: 各要素にインデックス値を書き込み
                // 非常に軽量（書き込みのみ）
                let num_elements = self.compute_num_elements(node);
                // ArangeもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln() + MEMORY_ACCESS_COST.ln() + lowering_penalty
            }
            GraphOp::Cast { .. } => {
                // 型変換: 各要素をキャスト
                // 非常に軽量（読み込み + キャスト + 書き込み）
                let num_elements = self.compute_num_elements(node);
                // CastもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln() + lowering_penalty
            }
            GraphOp::Real | GraphOp::Imag => {
                // 複素数から実部/虚部を抽出
                // 読み込み + 書き込み（stride 2でのアクセス）
                let num_elements = self.compute_num_elements(node);
                // Real/ImagもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln() + (2.0 * MEMORY_ACCESS_COST).ln() + lowering_penalty
            }
            GraphOp::ComplexFromParts => {
                // 実部と虚部から複素数を構築
                // 2つの入力を読み込み + インターリーブして書き込み
                let num_elements = self.compute_num_elements(node);
                // ComplexFromPartsもKernelにloweringされるべきなのでペナルティを追加
                let lowering_penalty = KERNEL_LAUNCH_OVERHEAD.ln();
                num_elements.ln() + (3.0 * MEMORY_ACCESS_COST).ln() + lowering_penalty
            }
            GraphOp::Kernel { ast, .. } => {
                // Kernel関数のコスト計算
                // KernelノードはLoweringSuggesterによって元の演算から変換されたもの
                //
                // ASTの内容に基づいてコストを推定する
                // これにより、AST最適化の効果がグラフのコストに反映される
                let ast_estimator = AstSimpleCostEstimator::new();

                match ast {
                    AstNode::Function { .. } | AstNode::Kernel { .. } | AstNode::Program { .. } => {
                        // Function/Kernel/Programの場合、ASTのコストを使用
                        let ast_cost = ast_estimator.estimate(ast);
                        // Kernelノードは複数のグラフノードを1つにまとめるため、
                        // カーネル起動オーバーヘッドの削減効果を反映させる
                        // カーネル起動オーバーヘッド = ln(1000) ≈ 6.9
                        // さらに大きな割引を適用してlowering/融合を強く優先させる
                        ast_cost - 3.0 * KERNEL_LAUNCH_OVERHEAD.ln()
                    }
                    _ => {
                        // その他のAST（Block等）の場合、要素数を考慮
                        let num_elements = self.compute_num_elements(node);
                        let ast_cost = ast_estimator.estimate(ast);
                        // 要素数とASTコストを組み合わせる
                        // ただし、ASTがループを含む場合は要素数が既に考慮されている可能性がある
                        if Self::ast_has_loop(ast) {
                            ast_cost - 3.0 * KERNEL_LAUNCH_OVERHEAD.ln()
                        } else {
                            num_elements.ln() + ast_cost - 3.0 * KERNEL_LAUNCH_OVERHEAD.ln()
                        }
                    }
                }
            }
            GraphOp::ProgramRoot { ast, .. } => {
                // ProgramRootノードのコスト = 含まれるProgramのコスト
                // ProgramRootはグラフのルートでProgramを保持する
                let ast_estimator = AstSimpleCostEstimator::new();
                ast_estimator.estimate(ast)
            }
        }
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

    /// ASTにループが含まれているかをチェック
    fn ast_has_loop(ast: &AstNode) -> bool {
        match ast {
            AstNode::Range { .. } => true,
            AstNode::Block { statements, .. } => statements.iter().any(Self::ast_has_loop),
            AstNode::Function { body, .. } => Self::ast_has_loop(body),
            AstNode::Program { functions, .. } => functions.iter().any(Self::ast_has_loop),
            _ => false,
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
        // ノード数ペナルティの計算時、出力Buffer（Kernel srcに含まれる出力バッファ）を除外
        // 出力Bufferは名前が "output" で始まる
        let node_count = nodes
            .iter()
            .filter(|n| !matches!(&n.op, GraphOp::Buffer { name } if name.starts_with("output")))
            .count();
        let mut log_costs = Vec::new();

        // カーネル数をカウント（カーネル起動オーバーヘッド計算用）
        // Kernel(Function)とlowering対象のノード（FusedElementwiseReduceなど）の両方をカウント
        // これにより、lowering前後でカーネルオーバーヘッドが変わらず、
        // loweringが進むようになる
        let mut kernel_count = 0;
        let mut has_custom_program = false;

        for node in &nodes {
            let log_base_cost = self.node_base_cost(node);
            log_costs.push(log_base_cost);

            // カーネルとしてカウントするノード
            match &node.op {
                GraphOp::Kernel { ast, .. } => match ast {
                    crate::ast::AstNode::Function { .. } => kernel_count += 1,
                    crate::ast::AstNode::Program { .. } => has_custom_program = true,
                    _ => {}
                },
                // lowering対象のノードもカーネルとしてカウント
                // これにより、lowering前後でオーバーヘッドが変わらない
                GraphOp::FusedElementwiseReduce { .. }
                | GraphOp::FusedElementwiseCumulative { .. }
                | GraphOp::FusedElementwise { .. }
                | GraphOp::Reduce { .. }
                | GraphOp::Cumulative { .. }
                | GraphOp::Contiguous
                | GraphOp::Pad { .. }
                | GraphOp::Slice { .. }
                | GraphOp::Concat { .. }
                | GraphOp::Rand
                | GraphOp::Arange
                | GraphOp::Cast { .. }
                | GraphOp::Real
                | GraphOp::Imag
                | GraphOp::ComplexFromParts => {
                    kernel_count += 1;
                }
                // Elementwiseもカーネルとしてカウント
                // これにより、loweringされていないElementwiseにペナルティが適用され、
                // 複数出力がある場合もすべてloweringされるようになる
                GraphOp::Elementwise { .. } => {
                    kernel_count += 1;
                }
                // Input, Const, Viewはカーネルではない
                _ => {}
            }
        }

        // カーネル起動オーバーヘッド
        // Kernel(Program)は1つのプログラムとして扱われるため、1回分のオーバーヘッド
        let kernel_count_f32 = if has_custom_program {
            1.0 // Programは全体で1つのカーネル起動として扱う
        } else if kernel_count > 0 {
            kernel_count as f32
        } else {
            graph.outputs().len() as f32
        };
        let log_kernel_overhead = kernel_count_f32.ln() + KERNEL_LAUNCH_OVERHEAD.ln();

        // すべてのコストを合計
        log_costs.push(log_kernel_overhead);
        let log_base_cost = log_sum_exp_iter(log_costs);

        // ノード数のペナルティを対数スケールで直接加算
        // final_cost = log_base_cost + penalty_coefficient * node_count
        // これは元のスケールで cost = base_cost * exp(penalty_coefficient * node_count)
        let penalty = self.node_count_penalty * node_count as f32;

        // 複数のKernel(Function)がある場合、強いペナルティを追加
        // これにより、単一のKernel(Program)への収束を強く優先する
        // kernel_count >= 2の場合：マージによりコストが大幅に下がるようにする
        let merge_penalty = if !has_custom_program && kernel_count >= 2 {
            // 2つ以上のカーネルがある場合、大きなペナルティを追加
            // これによりKernelMergeSuggesterの提案が採用されやすくなる
            KERNEL_LAUNCH_OVERHEAD.ln() * (kernel_count as f32 - 1.0)
        } else {
            0.0
        };

        // ProgramRoot.srcに入力Bufferがある場合、ペナルティを追加
        // これにより、ProgramRootBufferAbsorptionSuggesterの適用後にコストが下がる
        // 直接Buffer、またはView→Buffer(input)のパターンを検出（1レベルのみ）
        let sink_buffer_penalty = if let Some(sink) = graph.program_root() {
            let input_buffer_count = sink
                .src
                .iter()
                .filter(|s| {
                    // 直接入力Buffer
                    if matches!(&s.op, GraphOp::Buffer { name } if !name.starts_with("output")) {
                        return true;
                    }
                    // View→Buffer(input)パターン（再帰なし、1レベルのみ）
                    if matches!(&s.op, GraphOp::View(_)) {
                        return s.src.iter().any(|src| {
                            matches!(&src.op, GraphOp::Buffer { name } if !name.starts_with("output"))
                        });
                    }
                    false
                })
                .count();
            // 入力Bufferがあると、グラフが整理されていないとみなしてペナルティ
            // ペナルティを強くしてProgramRootBufferAbsorptionが選ばれやすくする
            2.0 * KERNEL_LAUNCH_OVERHEAD.ln() * input_buffer_count as f32
        } else {
            0.0
        };

        log_base_cost + penalty + merge_penalty + sink_buffer_penalty
    }
}

/// Lowering専用のコスト推定器
///
/// グラフノードをKernelノードに変換し、最終的に単一のProgramRootノードに
/// 集約することを強く優先します。
///
/// # コスト計算
/// - 非KernelノードGraphOp（Elementwise, Reduce等）に強いペナルティ
/// - 複数のKernel(Function)ノードにペナルティ
/// - 単一のProgramRootノードを持つグラフを強く優先
pub struct LoweringCostEstimator {
    /// 非Kernelノードあたりのペナルティ（対数スケール）
    non_custom_penalty: f32,
    /// Kernel(Function)あたりのペナルティ（対数スケール）
    function_penalty: f32,
}

impl LoweringCostEstimator {
    pub fn new() -> Self {
        Self {
            // 非KernelノードにはLoweringSuggesterを適用すべき
            // 非常に強いペナルティを設定し、Loweringを常に優先する
            // 元の値: KERNEL_LAUNCH_OVERHEAD.ln() * 2.0 ≈ 13.8
            // 新しい値: AST costを上回るように十分大きく設定
            non_custom_penalty: 100.0,
            // Kernel(Function)はProgramRootに吸収されるべき
            // こちらは小さめに設定して、Loweringの結果がペナルティで打ち消されないようにする
            function_penalty: 1.0,
        }
    }

    /// 非Kernelノードペナルティを設定
    pub fn with_non_custom_penalty(mut self, penalty: f32) -> Self {
        self.non_custom_penalty = penalty;
        self
    }

    /// Kernel(Function)ペナルティを設定
    pub fn with_function_penalty(mut self, penalty: f32) -> Self {
        self.function_penalty = penalty;
        self
    }

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

impl Default for LoweringCostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCostEstimator for LoweringCostEstimator {
    fn estimate(&self, graph: &Graph) -> f32 {
        let nodes = self.collect_all_nodes(graph);
        let ast_estimator = AstSimpleCostEstimator::new();

        let mut non_custom_count = 0;
        let mut custom_function_count = 0;
        let mut has_program_root = false;
        let mut total_ast_cost = f32::NEG_INFINITY;

        for node in &nodes {
            match &node.op {
                // 無視するノード（入力・メタデータのみ）
                GraphOp::Buffer { .. } | GraphOp::Const(_) | GraphOp::ComplexConst { .. } => {}
                GraphOp::View(_) => {}

                // Kernelノード
                GraphOp::Kernel { ast, .. } => match ast {
                    AstNode::Function { .. } => {
                        custom_function_count += 1;
                        let ast_cost = ast_estimator.estimate(ast);
                        total_ast_cost = log_sum_exp(total_ast_cost, ast_cost);
                    }
                    AstNode::Program { .. } => {
                        // Kernel(Program)はKernelMergeSuggesterによって生成される
                        let ast_cost = ast_estimator.estimate(ast);
                        total_ast_cost = log_sum_exp(total_ast_cost, ast_cost);
                    }
                    _ => {}
                },

                // ProgramRootノード（目標状態）
                GraphOp::ProgramRoot { ast, .. } => {
                    has_program_root = true;
                    let ast_cost = ast_estimator.estimate(ast);
                    total_ast_cost = log_sum_exp(total_ast_cost, ast_cost);
                }

                // Lowering対象のノード（強いペナルティ）
                _ => {
                    non_custom_count += 1;
                }
            }
        }

        // ベースコスト
        let base_cost = if total_ast_cost == f32::NEG_INFINITY {
            1.0
        } else {
            total_ast_cost
        };

        // ペナルティ計算
        // 1. 非Kernelノードへの強いペナルティ
        let non_custom_penalty = self.non_custom_penalty * non_custom_count as f32;

        // 2. Kernel(Function)へのペナルティ（ProgramRootがない場合のみ）
        let function_penalty = if !has_program_root && custom_function_count > 0 {
            self.function_penalty * custom_function_count as f32
        } else {
            0.0
        };

        // 3. ProgramRootがある場合は大きな報酬（負のペナルティ）
        let program_root_reward = if has_program_root {
            -KERNEL_LAUNCH_OVERHEAD.ln() * 3.0
        } else {
            0.0
        };

        base_cost + non_custom_penalty + function_penalty + program_root_reward
    }
}

/// KernelMerge専用のコスト推定器
///
/// 複数のKernel(Function)を1つのKernel(Program)にマージすることを強く優先します。
/// グラフ最適化の第2フェーズ（カーネルマージ）で使用します。
///
/// # コスト計算
/// - Kernel(Function)の数に強いペナルティを与える
/// - Kernel(Program)を持つグラフを強く優先する
/// - その他のノードは無視（既にlowering済みの前提）
pub struct KernelMergeCostEstimator {
    /// Kernel(Function)あたりのペナルティ（対数スケール）
    function_penalty: f32,
}

impl KernelMergeCostEstimator {
    pub fn new() -> Self {
        Self {
            // 各Kernel(Function)に大きなペナルティを与える
            // これにより、複数FunctionをProgramにマージすることが強く優先される
            function_penalty: 10.0,
        }
    }

    /// Kernel(Function)ペナルティを設定
    pub fn with_function_penalty(mut self, penalty: f32) -> Self {
        self.function_penalty = penalty;
        self
    }

    fn collect_all_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const std::ffi::c_void>,
            nodes: &mut Vec<GraphNode>,
        ) {
            let ptr = std::ptr::from_ref(node) as *const std::ffi::c_void;
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

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

impl Default for KernelMergeCostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCostEstimator for KernelMergeCostEstimator {
    fn estimate(&self, graph: &Graph) -> f32 {
        let nodes = self.collect_all_nodes(graph);
        let ast_estimator = AstSimpleCostEstimator::new();

        let mut custom_function_count = 0;
        let mut has_custom_program = false;
        let mut total_ast_cost = f32::NEG_INFINITY; // 対数スケールで0

        for node in &nodes {
            if let GraphOp::Kernel { ast, .. } = &node.op {
                match ast {
                    AstNode::Function { .. } => {
                        custom_function_count += 1;
                        let ast_cost = ast_estimator.estimate(ast);
                        total_ast_cost = log_sum_exp(total_ast_cost, ast_cost);
                    }
                    AstNode::Program { .. } => {
                        has_custom_program = true;
                        let ast_cost = ast_estimator.estimate(ast);
                        total_ast_cost = log_sum_exp(total_ast_cost, ast_cost);
                    }
                    _ => {}
                }
            }
        }

        // ASTコストがない（Kernelノードがない）場合はデフォルト値
        if total_ast_cost == f32::NEG_INFINITY {
            return 1.0;
        }

        // マージ状態に基づくペナルティをASTコストに加算
        // 対数スケール: log(ast_cost * penalty_factor) = log(ast_cost) + log(penalty_factor)
        let merge_penalty = if has_custom_program {
            // Programがあれば、追加のFunctionがあってもペナルティを軽減
            // （既にマージが始まっている状態）
            (1.0 + custom_function_count as f32 * 0.1).ln()
        } else if custom_function_count >= 2 {
            // 2つ以上のFunctionがある：マージ可能
            // Functionの数に応じてペナルティ
            (custom_function_count as f32 * self.function_penalty).ln()
        } else {
            // 0-1個のFunction：マージ不要または不可能
            0.0 // ln(1) = 0
        };

        total_ast_cost + merge_penalty
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
        let a = graph.input("a", DType::F32, vec![10, 20]);

        let b = graph.input("b", DType::F32, vec![10, 20]);

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
        let a1 = graph1.input("a", DType::F32, vec![10]);
        let b1 = graph1.input("b", DType::F32, vec![10]);
        graph1.output("c", a1 + b1);

        // 大きいグラフ
        let mut graph2 = Graph::new();
        let a2 = graph2.input("a", DType::F32, vec![1000]);
        let b2 = graph2.input("b", DType::F32, vec![1000]);
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
        let a1 = graph1.input("a", DType::F32, vec![100]);
        let b1 = graph1.input("b", DType::F32, vec![100]);
        graph1.output("c", a1 + b1);

        // 多いノード (5ノード: a, b, c, d, e)
        let mut graph2 = Graph::new();
        let a2 = graph2.input("a", DType::F32, vec![100]);
        let b2 = graph2.input("b", DType::F32, vec![100]);
        let c2 = graph2.input("c", DType::F32, vec![100]);
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
        let a = graph.input("a", DType::F32, vec![100]);
        let b = graph.input("b", DType::F32, vec![100]);
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
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        let cost = estimator.estimate(&graph);
        // コストは正の値であるべき
        assert!(cost > 0.0);
    }
}
