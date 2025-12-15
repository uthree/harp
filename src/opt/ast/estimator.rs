use super::AstCostEstimator;
use crate::ast::AstNode;
use crate::opt::cost_utils::{log_sum_exp, log_sum_exp_iter};

/// 簡単なコスト推定器
///
/// **重要**: このコスト推定器は対数スケール（log(CPUサイクル数)）でコストを返します。
/// 実際のサイクル数が必要な場合は、`result.exp()`を使用してください。
///
/// コストの単位は概ねCPUサイクル数を想定しています。
/// 実際のレイテンシは以下を参考にしています：
/// - 整数演算（加算/減算/ビット演算/シフト）: 1サイクル
/// - 整数乗算: 3サイクル
/// - 整数除算/剰余: 10-40サイクル
/// - 浮動小数点加算/減算: 3-4サイクル
/// - 浮動小数点乗算: 4-5サイクル
/// - 浮動小数点除算: 10-15サイクル
/// - 平方根: 10-20サイクル
/// - 超越関数（sin, exp, log）: 20-100サイクル
/// - メモリアクセス（L1キャッシュ）: 4サイクル
///
/// # 対数スケールの利点
/// - 大きな値を扱う際の数値的安定性
/// - 乗算が加算に変換されるため計算が簡潔
/// - オーバーフロー/アンダーフローのリスクが低減
///
/// # ノード数ペナルティ
/// ノード数に比例するペナルティ項を追加することで、ループ展開などで
/// ノード数が爆発する変換を抑制できます。ペナルティは対数スケールで
/// 直接加算されるため、元のスケールでは乗算的な効果があります。
#[derive(Clone, Debug)]
pub struct SimpleCostEstimator {
    /// ノード数あたりのペナルティ係数（対数スケール）
    pub node_count_penalty: f32,
    /// ループのオーバーヘッド（ループカウンタのインクリメント、比較、分岐）
    pub overhead_per_loop: f32,
    /// メモリアクセスのコスト（L1キャッシュヒット想定）
    pub memory_access_cost: f32,
    /// 関数呼び出しのオーバーヘッド（スタックフレームの設定、レジスタ退避など）
    pub function_call_overhead: f32,
    /// 関数定義のオーバーヘッド（プロローグ/エピローグ生成、シンボルテーブルエントリなど）
    pub function_definition_overhead: f32,
    /// 同期バリアのコスト（スレッド間の同期待ち）
    pub barrier_cost: f32,
    /// 条件分岐のオーバーヘッド（分岐予測ミスを考慮）
    pub branch_overhead: f32,
    /// 比較演算のコスト（整数比較）
    pub comparison_cost: f32,
    /// 連続ループの境界が揃っている場合のボーナス（融合可能性への報酬）
    pub loop_fusion_bonus: f32,
    /// GPUカーネル起動オーバーヘッド（CPUサイクル相当）
    pub kernel_launch_overhead: f32,
    /// GPU並列化の効果が出る最小グリッドサイズ
    pub gpu_parallel_min_grid_size: f32,
    /// GPU並列化による最大スピードアップ係数（対数スケール）
    pub gpu_parallel_max_speedup_log: f32,
    /// メモリバンド幅制限が効き始めるスレッド数
    pub gpu_memory_bandwidth_threshold: f32,
}

impl Default for SimpleCostEstimator {
    fn default() -> Self {
        Self {
            node_count_penalty: 0.01,
            overhead_per_loop: 2.0,
            memory_access_cost: 4.0,
            function_call_overhead: 10.0,
            function_definition_overhead: 50.0,
            barrier_cost: 100.0,
            branch_overhead: 5.0,
            comparison_cost: 1.0,
            loop_fusion_bonus: 50.0,
            kernel_launch_overhead: 1000.0,
            gpu_parallel_min_grid_size: 512.0,
            gpu_parallel_max_speedup_log: 4.5,
            gpu_memory_bandwidth_threshold: 10000.0,
        }
    }
}

impl SimpleCostEstimator {
    /// 新しいコスト推定器を作成
    pub fn new() -> Self {
        Self::default()
    }

    /// ノード数ペナルティ係数を設定
    pub fn with_node_count_penalty(mut self, penalty: f32) -> Self {
        self.node_count_penalty = penalty;
        self
    }

    /// ループオーバーヘッドを設定
    pub fn with_overhead_per_loop(mut self, overhead: f32) -> Self {
        self.overhead_per_loop = overhead;
        self
    }

    /// メモリアクセスコストを設定
    pub fn with_memory_access_cost(mut self, cost: f32) -> Self {
        self.memory_access_cost = cost;
        self
    }

    /// 関数呼び出しオーバーヘッドを設定
    pub fn with_function_call_overhead(mut self, overhead: f32) -> Self {
        self.function_call_overhead = overhead;
        self
    }

    /// 関数定義オーバーヘッドを設定
    pub fn with_function_definition_overhead(mut self, overhead: f32) -> Self {
        self.function_definition_overhead = overhead;
        self
    }

    /// バリアコストを設定
    pub fn with_barrier_cost(mut self, cost: f32) -> Self {
        self.barrier_cost = cost;
        self
    }

    /// 分岐オーバーヘッドを設定
    pub fn with_branch_overhead(mut self, overhead: f32) -> Self {
        self.branch_overhead = overhead;
        self
    }

    /// 比較演算コストを設定
    pub fn with_comparison_cost(mut self, cost: f32) -> Self {
        self.comparison_cost = cost;
        self
    }

    /// ループ融合ボーナスを設定
    pub fn with_loop_fusion_bonus(mut self, bonus: f32) -> Self {
        self.loop_fusion_bonus = bonus;
        self
    }

    /// カーネル起動オーバーヘッドを設定
    pub fn with_kernel_launch_overhead(mut self, overhead: f32) -> Self {
        self.kernel_launch_overhead = overhead;
        self
    }

    /// GPU並列化最小グリッドサイズを設定
    pub fn with_gpu_parallel_min_grid_size(mut self, size: f32) -> Self {
        self.gpu_parallel_min_grid_size = size;
        self
    }

    /// GPU並列化最大スピードアップを設定
    pub fn with_gpu_parallel_max_speedup_log(mut self, speedup: f32) -> Self {
        self.gpu_parallel_max_speedup_log = speedup;
        self
    }

    /// GPUメモリバンド幅しきい値を設定
    pub fn with_gpu_memory_bandwidth_threshold(mut self, threshold: f32) -> Self {
        self.gpu_memory_bandwidth_threshold = threshold;
        self
    }

    /// 2つのASTノードが構造的に等しいかチェック（境界比較用）
    fn ast_equal(a: &AstNode, b: &AstNode) -> bool {
        match (a, b) {
            (AstNode::Const(l1), AstNode::Const(l2)) => l1 == l2,
            (AstNode::Var(n1), AstNode::Var(n2)) => n1 == n2,
            (AstNode::Add(a1, b1), AstNode::Add(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::Mul(a1, b1), AstNode::Mul(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            _ => false,
        }
    }

    /// Block内の連続するRangeの境界が揃っている数をカウント
    fn count_fusable_loop_pairs(statements: &[AstNode]) -> usize {
        if statements.len() < 2 {
            return 0;
        }

        let mut count = 0;
        for i in 0..statements.len() - 1 {
            if let (
                AstNode::Range {
                    start: start1,
                    step: step1,
                    stop: stop1,
                    ..
                },
                AstNode::Range {
                    start: start2,
                    step: step2,
                    stop: stop2,
                    ..
                },
            ) = (&statements[i], &statements[i + 1])
                && Self::ast_equal(start1, start2)
                && Self::ast_equal(step1, step2)
                && Self::ast_equal(stop1, stop2)
            {
                count += 1;
            }
        }
        count
    }

    /// ASTノードの総数をカウント（コンパイル時間・メモリ使用量の指標）
    fn count_nodes(ast: &AstNode) -> usize {
        let children_count: usize = match ast {
            AstNode::Add(l, r)
            | AstNode::Mul(l, r)
            | AstNode::Max(l, r)
            | AstNode::Rem(l, r)
            | AstNode::Idiv(l, r)
            | AstNode::BitwiseAnd(l, r)
            | AstNode::BitwiseOr(l, r)
            | AstNode::BitwiseXor(l, r)
            | AstNode::LeftShift(l, r)
            | AstNode::RightShift(l, r) => Self::count_nodes(l) + Self::count_nodes(r),
            AstNode::Recip(n)
            | AstNode::Sqrt(n)
            | AstNode::Log2(n)
            | AstNode::Exp2(n)
            | AstNode::Sin(n)
            | AstNode::BitwiseNot(n) => Self::count_nodes(n),
            AstNode::Cast(n, _) => Self::count_nodes(n),
            AstNode::Load { ptr, offset, .. } => Self::count_nodes(ptr) + Self::count_nodes(offset),
            AstNode::Store { ptr, offset, value } => {
                Self::count_nodes(ptr) + Self::count_nodes(offset) + Self::count_nodes(value)
            }
            AstNode::Assign { value, .. } => Self::count_nodes(value),
            AstNode::Range {
                start,
                step,
                stop,
                body,
                ..
            } => {
                Self::count_nodes(start)
                    + Self::count_nodes(step)
                    + Self::count_nodes(stop)
                    + Self::count_nodes(body)
            }
            AstNode::Block { statements, .. } => statements.iter().map(Self::count_nodes).sum(),
            AstNode::Call { args, .. } => args.iter().map(Self::count_nodes).sum(),
            AstNode::Return { value } => Self::count_nodes(value),
            AstNode::Function { body, .. } => Self::count_nodes(body),
            AstNode::Kernel {
                body,
                default_grid_size,
                default_thread_group_size,
                ..
            } => {
                Self::count_nodes(body)
                    + default_grid_size
                        .iter()
                        .map(|g| Self::count_nodes(g))
                        .sum::<usize>()
                    + default_thread_group_size
                        .iter()
                        .map(|t| Self::count_nodes(t))
                        .sum::<usize>()
            }
            AstNode::CallKernel {
                args,
                grid_size,
                thread_group_size,
                ..
            } => {
                args.iter().map(Self::count_nodes).sum::<usize>()
                    + grid_size
                        .iter()
                        .map(|g| Self::count_nodes(g))
                        .sum::<usize>()
                    + thread_group_size
                        .iter()
                        .map(|t| Self::count_nodes(t))
                        .sum::<usize>()
            }
            AstNode::Allocate { size, .. } => Self::count_nodes(size),
            AstNode::Deallocate { ptr } => Self::count_nodes(ptr),
            AstNode::Program { functions, .. } => functions.iter().map(Self::count_nodes).sum(),
            // 比較演算
            AstNode::Lt(a, b)
            | AstNode::Le(a, b)
            | AstNode::Gt(a, b)
            | AstNode::Ge(a, b)
            | AstNode::Eq(a, b)
            | AstNode::Ne(a, b) => Self::count_nodes(a) + Self::count_nodes(b),
            // 条件分岐
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                let base = Self::count_nodes(condition) + Self::count_nodes(then_body);
                if let Some(else_node) = else_body {
                    base + Self::count_nodes(else_node)
                } else {
                    base
                }
            }
            _ => 0, // Const, Var, Barrier, Rand, etc.
        };
        1 + children_count
    }

    /// ノードのベースコストを取得（log(CPUサイクル数)）
    fn base_cost(&self, ast: &AstNode) -> f32 {
        match ast {
            // ビット演算とシフト演算（高速）
            AstNode::BitwiseAnd(_, _) | AstNode::BitwiseOr(_, _) | AstNode::BitwiseXor(_, _) => {
                1.0_f32.ln()
            }
            AstNode::BitwiseNot(_) => 1.0_f32.ln(),
            AstNode::LeftShift(_, _) | AstNode::RightShift(_, _) => 1.0_f32.ln(),

            // 加算・減算（高速）
            AstNode::Add(_, _) => 3.0_f32.ln(),

            // 乗算（中速）
            AstNode::Mul(_, _) => 4.0_f32.ln(),

            // 除算・剰余（低速）
            AstNode::Idiv(_, _) => 25.0_f32.ln(),
            AstNode::Rem(_, _) => 25.0_f32.ln(),
            AstNode::Recip(_) => 14.0_f32.ln(),

            // 数学関数（低速）
            AstNode::Sqrt(_) => 15.0_f32.ln(),
            AstNode::Log2(_) => 40.0_f32.ln(),
            AstNode::Exp2(_) => 40.0_f32.ln(),
            AstNode::Sin(_) => 50.0_f32.ln(),

            // 比較・選択
            AstNode::Max(_, _) => 2.0_f32.ln(),

            // メモリアクセス
            AstNode::Load { count, .. } => (self.memory_access_cost * (*count as f32)).ln(),
            AstNode::Store { .. } => self.memory_access_cost.ln(),

            // 型変換（整数↔浮動小数点など）
            AstNode::Cast(_, _) => 2.0_f32.ln(),

            // 同期バリア（スレッド間の同期待ち）
            AstNode::Barrier => self.barrier_cost.ln(),

            // 乱数生成（比較的高コスト）
            AstNode::Rand => 10.0_f32.ln(),

            // メモリ確保/解放（高コスト）
            AstNode::Allocate { .. } => 100.0_f32.ln(),
            AstNode::Deallocate { .. } => 50.0_f32.ln(),

            // GPUカーネル呼び出し（非常に高いオーバーヘッド）
            AstNode::CallKernel { .. } => self.barrier_cost.ln(),

            // 比較演算（高速）
            AstNode::Lt(_, _)
            | AstNode::Le(_, _)
            | AstNode::Gt(_, _)
            | AstNode::Ge(_, _)
            | AstNode::Eq(_, _)
            | AstNode::Ne(_, _) => self.comparison_cost.ln(),

            // 条件分岐（分岐予測オーバーヘッドを含む）
            AstNode::If { .. } => self.branch_overhead.ln(),

            // その他（変数参照、定数など）
            _ => f32::NEG_INFINITY, // log(0) = -∞
        }
    }
}

impl AstCostEstimator for SimpleCostEstimator {
    fn estimate(&self, ast: &AstNode) -> f32 {
        let base_cost = self.base_cost(ast);

        // 子ノードのコストを再帰的に計算（対数スケール）
        let children_cost: f32 = match ast {
            AstNode::Add(l, r)
            | AstNode::Mul(l, r)
            | AstNode::Max(l, r)
            | AstNode::Rem(l, r)
            | AstNode::Idiv(l, r)
            | AstNode::BitwiseAnd(l, r)
            | AstNode::BitwiseOr(l, r)
            | AstNode::BitwiseXor(l, r)
            | AstNode::LeftShift(l, r)
            | AstNode::RightShift(l, r) => log_sum_exp(self.estimate(l), self.estimate(r)),
            AstNode::Recip(n)
            | AstNode::Sqrt(n)
            | AstNode::Log2(n)
            | AstNode::Exp2(n)
            | AstNode::Sin(n)
            | AstNode::BitwiseNot(n) => self.estimate(n),
            AstNode::Cast(n, _) => self.estimate(n),
            AstNode::Load { ptr, offset, .. } => {
                log_sum_exp(self.estimate(ptr), self.estimate(offset))
            }
            AstNode::Store { ptr, offset, value } => log_sum_exp(
                log_sum_exp(self.estimate(ptr), self.estimate(offset)),
                self.estimate(value),
            ),
            AstNode::Assign { value, .. } => self.estimate(value),
            AstNode::Range {
                start,
                step,
                stop,
                body,
                ..
            } => {
                // start, stop, stepが定数の場合は実際のループ回数を計算
                let loop_count = match (start.as_ref(), stop.as_ref(), step.as_ref()) {
                    (
                        AstNode::Const(start_lit),
                        AstNode::Const(stop_lit),
                        AstNode::Const(step_lit),
                    ) => {
                        // 整数リテラル（Isize または Usize）から値を取得
                        if let (Some(start_val), Some(stop_val), Some(step_val)) = (
                            start_lit.as_isize(),
                            stop_lit.as_isize(),
                            step_lit.as_isize(),
                        ) {
                            if step_val > 0 {
                                // 正の方向のループ
                                let iterations = (stop_val - start_val + step_val - 1) / step_val;
                                iterations.max(0) as f32
                            } else if step_val < 0 {
                                // 負の方向のループ
                                let iterations =
                                    (start_val - stop_val - step_val - 1) / (-step_val);
                                iterations.max(0) as f32
                            } else {
                                // step_val == 0 の場合は無限ループになるので、デフォルト値を使用
                                log::warn!("detected infinity loop");
                                100.0
                            }
                        } else {
                            // 整数リテラルではない（F32など）場合はデフォルト値
                            100.0
                        }
                    }
                    _ => {
                        // ループ回数が不明な場合は100回と推定
                        100.0
                    }
                };
                // 対数スケールでの乗算: log(a * b) = log(a) + log(b)
                // (body_cost + step_cost + stop_cost + overhead) * loop_count
                // → log(loop_count) + log_sum_exp(body_cost, step_cost, stop_cost, log(overhead))
                let log_loop_count = loop_count.ln();
                let per_iteration_cost = log_sum_exp_iter(vec![
                    self.estimate(body),
                    self.estimate(step),
                    self.estimate(stop),
                    self.overhead_per_loop.ln(),
                ]);
                log_sum_exp(self.estimate(start), log_loop_count + per_iteration_cost)
            }
            AstNode::Block { statements, .. } => {
                let statements_cost = log_sum_exp_iter(statements.iter().map(|s| self.estimate(s)));

                // 連続するループの境界が揃っている場合、融合可能としてボーナスを与える
                let fusable_pairs = Self::count_fusable_loop_pairs(statements);
                if fusable_pairs > 0 {
                    // ボーナスを減算（対数スケールなので、低いコストが良い）
                    let bonus = (fusable_pairs as f32 * self.loop_fusion_bonus).ln();
                    statements_cost - bonus
                } else {
                    statements_cost
                }
            }
            AstNode::Call { args, .. } => {
                // 関数呼び出しは引数の評価コスト + 呼び出しオーバーヘッド
                let args_cost = log_sum_exp_iter(args.iter().map(|a| self.estimate(a)));
                log_sum_exp(args_cost, self.function_call_overhead.ln())
            }
            // 比較演算
            AstNode::Lt(a, b)
            | AstNode::Le(a, b)
            | AstNode::Gt(a, b)
            | AstNode::Ge(a, b)
            | AstNode::Eq(a, b)
            | AstNode::Ne(a, b) => log_sum_exp(self.estimate(a), self.estimate(b)),
            // 条件分岐
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                // 条件式 + then節のコスト
                let condition_cost = self.estimate(condition);
                let then_cost = self.estimate(then_body);

                // else節がある場合はそのコストも加算
                // 実行時にはどちらか一方のみ実行されるが、
                // コスト推定では最悪ケースを考慮
                let branch_cost = if let Some(else_node) = else_body {
                    let else_cost = self.estimate(else_node);
                    // どちらか大きい方を採用（最悪ケース）
                    log_sum_exp(then_cost, else_cost)
                } else {
                    then_cost
                };

                log_sum_exp(condition_cost, branch_cost)
            }
            AstNode::Return { value } => self.estimate(value),
            AstNode::Allocate { size, .. } => self.estimate(size),
            AstNode::Deallocate { ptr } => self.estimate(ptr),
            AstNode::CallKernel {
                args,
                grid_size,
                thread_group_size,
                ..
            } => {
                // GPUカーネル呼び出しは引数の評価 + grid/threadの設定
                let args_cost = log_sum_exp_iter(args.iter().map(|a| self.estimate(a)));
                let grid_cost = log_sum_exp_iter(grid_size.iter().map(|g| self.estimate(g)));
                let thread_cost =
                    log_sum_exp_iter(thread_group_size.iter().map(|t| self.estimate(t)));
                log_sum_exp_iter(vec![args_cost, grid_cost, thread_cost])
            }
            AstNode::Function { body, .. } => {
                // 関数本体のコスト + 関数定義オーバーヘッド
                // 関数定義自体にもコストがかかる（プロローグ/エピローグ、スタックフレーム管理など）
                log_sum_exp(self.estimate(body), self.function_definition_overhead.ln())
            }
            AstNode::Kernel {
                body,
                default_grid_size,
                default_thread_group_size,
                ..
            } => {
                // GPUカーネルのコスト計算
                // カーネル本体は各スレッドで並列実行されるため、
                // 並列化によるスピードアップを考慮する
                let body_cost = self.estimate(body);

                // grid_sizeから総スレッド数（並列度）を推定
                let grid_elements: f32 = default_grid_size
                    .iter()
                    .filter_map(|g| match g.as_ref() {
                        AstNode::Const(lit) => lit.as_usize().map(|v| v as f32),
                        _ => None,
                    })
                    .product();
                let thread_elements: f32 = default_thread_group_size
                    .iter()
                    .filter_map(|t| match t.as_ref() {
                        AstNode::Const(lit) => lit.as_usize().map(|v| v as f32),
                        _ => None,
                    })
                    .product();
                let total_threads = if grid_elements > 0.0 && thread_elements > 0.0 {
                    grid_elements * thread_elements
                } else if grid_elements > 0.0 {
                    grid_elements
                } else {
                    // 不明な場合は控えめな並列度を想定
                    1024.0
                };

                // スレッド数に基づく実効スピードアップの計算
                let parallel_bonus = if total_threads >= self.gpu_parallel_min_grid_size {
                    // 理想的なスピードアップ = log(threads)
                    let ideal_speedup = total_threads.ln();

                    // メモリバンド幅による効率低下
                    // スレッド数が増えるほどメモリアクセスがボトルネックになる
                    // efficiency = 1 / (1 + log(threads / threshold)) for threads > threshold
                    let memory_efficiency = if total_threads > self.gpu_memory_bandwidth_threshold {
                        1.0 / (1.0 + (total_threads / self.gpu_memory_bandwidth_threshold).ln())
                    } else {
                        1.0
                    };

                    // 実効スピードアップ = 理想スピードアップ × メモリ効率
                    // ただし上限あり
                    (ideal_speedup * memory_efficiency).min(self.gpu_parallel_max_speedup_log)
                } else if total_threads > 1.0 {
                    // 小規模でも多少の並列化効果はある
                    // ただし効率は低い（起動オーバーヘッドが相対的に大きい）
                    (total_threads.ln() * 0.5).max(0.0)
                } else {
                    // 単一スレッド: 並列化なし
                    0.0
                };

                // grid/thread設定の評価コスト（通常は定数なので無視できる）
                let grid_cost =
                    log_sum_exp_iter(default_grid_size.iter().map(|g| self.estimate(g)));
                let thread_cost =
                    log_sum_exp_iter(default_thread_group_size.iter().map(|t| self.estimate(t)));

                // 最終コスト = body_cost - parallel_bonus + カーネル起動オーバーヘッド
                // 並列化ボーナスを減算することで、GPUカーネルが有利になる
                log_sum_exp_iter(vec![
                    body_cost - parallel_bonus,
                    grid_cost,
                    thread_cost,
                    self.kernel_launch_overhead.ln(),
                    self.function_definition_overhead.ln(),
                ])
            }
            AstNode::Program { functions } => {
                // 空のプログラムの場合は最小コストを返す
                if functions.is_empty() {
                    return 0.0; // log(1) = 0、最小コスト
                }

                // 関数/カーネルの数に基づいてペナルティを計算
                // カーネルが多いほどコストが高くなる（融合を促進）
                let kernel_count = functions
                    .iter()
                    .filter(|f| matches!(f, AstNode::Function { .. } | AstNode::Kernel { .. }))
                    .count();

                // 複数のカーネルがある場合、追加のペナルティ（1つは基準として引く）
                let non_entry_functions = kernel_count.saturating_sub(1);

                // すべての関数のコストの合計
                let functions_cost = log_sum_exp_iter(functions.iter().map(|f| self.estimate(f)));

                // 非エントリポイント関数の数に比例したペナルティを追加
                // 関数が多いほど、コードサイズが大きくなり、キャッシュ効率が悪化する
                let base_program_cost = if non_entry_functions > 0 {
                    let penalty =
                        (non_entry_functions as f32 * self.function_definition_overhead).ln();
                    log_sum_exp(functions_cost, penalty)
                } else {
                    functions_cost
                };

                // ノード数に基づくペナルティを対数スケールで直接加算
                // final_cost = base_cost + penalty_coefficient * node_count
                // これは元のスケールで cost = base_cost * exp(penalty_coefficient * node_count)
                let node_count = Self::count_nodes(ast);
                let node_penalty = self.node_count_penalty * node_count as f32;

                base_program_cost + node_penalty
            }
            _ => f32::NEG_INFINITY, // log(0)
        };

        log_sum_exp(base_cost, children_cost)
    }
}

impl SimpleCostEstimator {
    /// ASTのノード数を取得（デバッグ用公開メソッド）
    pub fn get_node_count(ast: &AstNode) -> usize {
        Self::count_nodes(ast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, Literal, Mutability, Scope, VarDecl, VarKind};

    #[test]
    fn test_function_inlining_reduces_cost() {
        let estimator = SimpleCostEstimator::new();

        // インライン展開前: 2つの関数（add_one + main）
        // fn add_one(x: Int) -> Int { return x + 1 }
        // fn main() -> Int { return add_one(5) }
        let add_one_func = AstNode::Function {
            name: Some("add_one".to_string()),
            params: vec![VarDecl {
                name: "x".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::Int,
            body: Box::new(AstNode::Return {
                value: Box::new(AstNode::Add(
                    Box::new(AstNode::Var("x".to_string())),
                    Box::new(AstNode::Const(Literal::Int(1))),
                )),
            }),
        };

        let main_with_call = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::Int,
            body: Box::new(AstNode::Return {
                value: Box::new(AstNode::Call {
                    name: "add_one".to_string(),
                    args: vec![AstNode::Const(Literal::Int(5))],
                }),
            }),
        };

        let program_before = AstNode::Program {
            functions: vec![add_one_func, main_with_call],
        };

        // インライン展開後: 1つの関数（mainのみ）
        // fn main() -> Int { return 5 + 1 }
        let main_inlined = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::Int,
            body: Box::new(AstNode::Return {
                value: Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Int(5))),
                    Box::new(AstNode::Const(Literal::Int(1))),
                )),
            }),
        };

        let program_after = AstNode::Program {
            functions: vec![main_inlined],
        };

        let cost_before = estimator.estimate(&program_before);
        let cost_after = estimator.estimate(&program_after);

        // インライン展開後の方がコストが低いはず
        // （関数定義オーバーヘッド + 関数呼び出しオーバーヘッドが削減される）
        assert!(
            cost_after < cost_before,
            "Expected cost after inlining ({}) to be less than before ({})",
            cost_after,
            cost_before
        );
    }

    #[test]
    fn test_more_functions_higher_cost() {
        let estimator = SimpleCostEstimator::new();

        // 1つの関数
        let single_func = AstNode::Program {
            functions: vec![AstNode::Function {
                name: Some("main".to_string()),
                params: vec![],
                return_type: DType::Int,
                body: Box::new(AstNode::Return {
                    value: Box::new(AstNode::Const(Literal::Int(1))),
                }),
            }],
        };

        // 2つの関数（同じ本体だが関数定義が多い）
        let two_funcs = AstNode::Program {
            functions: vec![
                AstNode::Function {
                    name: Some("helper".to_string()),
                    params: vec![],
                    return_type: DType::Int,
                    body: Box::new(AstNode::Return {
                        value: Box::new(AstNode::Const(Literal::Int(1))),
                    }),
                },
                AstNode::Function {
                    name: Some("main".to_string()),
                    params: vec![],
                    return_type: DType::Int,
                    body: Box::new(AstNode::Return {
                        value: Box::new(AstNode::Const(Literal::Int(1))),
                    }),
                },
            ],
        };

        let cost_single = estimator.estimate(&single_func);
        let cost_two = estimator.estimate(&two_funcs);

        // 関数が多い方がコストが高い（関数定義オーバーヘッドが追加される）
        assert!(
            cost_two > cost_single,
            "Expected cost with two functions ({}) to be greater than one ({})",
            cost_two,
            cost_single
        );
    }

    #[test]
    fn test_call_overhead() {
        let estimator = SimpleCostEstimator::new();

        // 直接計算: 5 + 1
        let direct = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(5))),
            Box::new(AstNode::Const(Literal::Int(1))),
        );

        // 関数呼び出し経由: call add_one(5)
        let via_call = AstNode::Call {
            name: "add_one".to_string(),
            args: vec![AstNode::Const(Literal::Int(5))],
        };

        let cost_direct = estimator.estimate(&direct);
        let cost_via_call = estimator.estimate(&via_call);

        // 関数呼び出しにはオーバーヘッドがあるので、直接計算より高コスト
        assert!(
            cost_via_call > cost_direct,
            "Expected call overhead to increase cost: {} > {}",
            cost_via_call,
            cost_direct
        );
    }

    #[test]
    fn test_void_function_inlining_reduces_cost() {
        let estimator = SimpleCostEstimator::new();

        // インライン展開前: 副作用関数を呼び出し
        let write_value_func = AstNode::Function {
            name: Some("write_value".to_string()),
            params: vec![
                VarDecl {
                    name: "ptr".to_string(),
                    dtype: DType::Ptr(Box::new(DType::Int)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "value".to_string(),
                    dtype: DType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Store {
                    ptr: Box::new(AstNode::Var("ptr".to_string())),
                    offset: Box::new(AstNode::Const(Literal::Int(0))),
                    value: Box::new(AstNode::Var("value".to_string())),
                }],
                scope: Box::new(Scope::new()),
            }),
        };

        let main_with_call = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Call {
                    name: "write_value".to_string(),
                    args: vec![
                        AstNode::Var("buffer".to_string()),
                        AstNode::Const(Literal::Int(42)),
                    ],
                }],
                scope: Box::new(Scope::new()),
            }),
        };

        let program_before = AstNode::Program {
            functions: vec![write_value_func, main_with_call],
        };

        // インライン展開後
        let main_inlined = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Store {
                    ptr: Box::new(AstNode::Var("buffer".to_string())),
                    offset: Box::new(AstNode::Const(Literal::Int(0))),
                    value: Box::new(AstNode::Const(Literal::Int(42))),
                }],
                scope: Box::new(Scope::new()),
            }),
        };

        let program_after = AstNode::Program {
            functions: vec![main_inlined],
        };

        let cost_before = estimator.estimate(&program_before);
        let cost_after = estimator.estimate(&program_after);

        // インライン展開後の方がコストが低い
        assert!(
            cost_after < cost_before,
            "Expected cost after void function inlining ({}) to be less than before ({})",
            cost_after,
            cost_before
        );
    }

    #[test]
    fn test_node_count() {
        // 単純なノード
        let const_node = AstNode::Const(Literal::Int(42));
        assert_eq!(SimpleCostEstimator::get_node_count(&const_node), 1);

        // 二項演算（3ノード: Add + 2つの定数）
        let add_node = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(1))),
            Box::new(AstNode::Const(Literal::Int(2))),
        );
        assert_eq!(SimpleCostEstimator::get_node_count(&add_node), 3);

        // ネストした演算（5ノード: Add + Mul + 3つの定数）
        let nested = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Const(Literal::Int(2))),
                Box::new(AstNode::Const(Literal::Int(3))),
            )),
            Box::new(AstNode::Const(Literal::Int(1))),
        );
        assert_eq!(SimpleCostEstimator::get_node_count(&nested), 5);
    }

    #[test]
    fn test_node_count_penalty() {
        let estimator = SimpleCostEstimator::new();

        // 小さいプログラム
        let small_program = AstNode::Program {
            functions: vec![AstNode::Function {
                name: Some("main".to_string()),
                params: vec![],
                return_type: DType::Int,
                body: Box::new(AstNode::Return {
                    value: Box::new(AstNode::Const(Literal::Int(1))),
                }),
            }],
        };

        let small_node_count = SimpleCostEstimator::get_node_count(&small_program);

        // 大きいプログラム
        // 多数のステートメントを持つBlock
        let mut statements = Vec::new();
        for i in 0..200 {
            statements.push(AstNode::Assign {
                var: format!("var_{}", i),
                value: Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Int(i as isize))),
                    Box::new(AstNode::Const(Literal::Int(1))),
                )),
            });
        }

        let large_program = AstNode::Program {
            functions: vec![AstNode::Function {
                name: Some("main".to_string()),
                params: vec![],
                return_type: DType::Tuple(vec![]),
                body: Box::new(AstNode::Block {
                    statements,
                    scope: Box::new(Scope::new()),
                }),
            }],
        };

        let large_node_count = SimpleCostEstimator::get_node_count(&large_program);
        assert!(large_node_count > small_node_count);

        let cost_small = estimator.estimate(&small_program);
        let cost_large = estimator.estimate(&large_program);

        // 大きいプログラムの方がコストが高い（ノード数ペナルティの効果）
        assert!(
            cost_large > cost_small,
            "Expected larger program cost ({}) to be greater than small program cost ({})",
            cost_large,
            cost_small
        );
    }

    #[test]
    fn test_if_node_cost() {
        let estimator = SimpleCostEstimator::new();

        // 単純なIf文: if (x < 10) { y }
        let if_node = AstNode::If {
            condition: Box::new(AstNode::Lt(
                Box::new(AstNode::Var("x".to_string())),
                Box::new(AstNode::Const(Literal::Int(10))),
            )),
            then_body: Box::new(AstNode::Var("y".to_string())),
            else_body: None,
        };

        let cost = estimator.estimate(&if_node);
        // コストは有限値であるべき
        assert!(
            cost.is_finite(),
            "If node cost should be finite, got: {}",
            cost
        );

        // 条件式だけのコストと比較（Ifはオーバーヘッドがある）
        let condition_only = AstNode::Lt(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(10))),
        );
        let condition_cost = estimator.estimate(&condition_only);

        assert!(
            cost > condition_cost,
            "If node cost ({}) should be greater than condition-only cost ({})",
            cost,
            condition_cost
        );
    }

    #[test]
    fn test_comparison_operators_cost() {
        let estimator = SimpleCostEstimator::new();

        // 各比較演算子のコストをテスト
        let lt = AstNode::Lt(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        let le = AstNode::Le(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        let gt = AstNode::Gt(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        let ge = AstNode::Ge(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        let eq = AstNode::Eq(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        let ne = AstNode::Ne(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );

        // 全ての比較演算子のコストが有限値であることを確認
        for (name, node) in [
            ("Lt", lt),
            ("Le", le),
            ("Gt", gt),
            ("Ge", ge),
            ("Eq", eq),
            ("Ne", ne),
        ] {
            let cost = estimator.estimate(&node);
            assert!(
                cost.is_finite(),
                "{} cost should be finite, got: {}",
                name,
                cost
            );
        }
    }

    #[test]
    fn test_nested_if_cost() {
        let estimator = SimpleCostEstimator::new();

        // ネストしたIf: if (a < b) { if (c < d) { x } }
        let nested_if = AstNode::If {
            condition: Box::new(AstNode::Lt(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            )),
            then_body: Box::new(AstNode::If {
                condition: Box::new(AstNode::Lt(
                    Box::new(AstNode::Var("c".to_string())),
                    Box::new(AstNode::Var("d".to_string())),
                )),
                then_body: Box::new(AstNode::Var("x".to_string())),
                else_body: None,
            }),
            else_body: None,
        };

        // 単純なIf
        let simple_if = AstNode::If {
            condition: Box::new(AstNode::Lt(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            )),
            then_body: Box::new(AstNode::Var("x".to_string())),
            else_body: None,
        };

        let nested_cost = estimator.estimate(&nested_if);
        let simple_cost = estimator.estimate(&simple_if);

        // ネストしたIfの方がコストが高い
        assert!(
            nested_cost > simple_cost,
            "Nested if cost ({}) should be greater than simple if cost ({})",
            nested_cost,
            simple_cost
        );
    }

    #[test]
    fn test_node_count_with_if() {
        // Ifノードを含むノード数のカウント
        let if_node = AstNode::If {
            condition: Box::new(AstNode::Lt(
                Box::new(AstNode::Var("x".to_string())),
                Box::new(AstNode::Const(Literal::Int(10))),
            )),
            then_body: Box::new(AstNode::Var("y".to_string())),
            else_body: None,
        };

        // If(1) + Lt(1) + Var(1) + Const(1) + Var(1) = 5
        let count = SimpleCostEstimator::get_node_count(&if_node);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_node_count_penalty_configurable() {
        // ペナルティなし
        let estimator_no_penalty = SimpleCostEstimator::new().with_node_count_penalty(0.0);
        // 高いペナルティ
        let estimator_high_penalty = SimpleCostEstimator::new().with_node_count_penalty(1.0);

        let program = AstNode::Program {
            functions: vec![AstNode::Function {
                name: Some("main".to_string()),
                params: vec![],
                return_type: DType::Int,
                body: Box::new(AstNode::Return {
                    value: Box::new(AstNode::Add(
                        Box::new(AstNode::Const(Literal::Int(1))),
                        Box::new(AstNode::Const(Literal::Int(2))),
                    )),
                }),
            }],
        };

        let cost_no_penalty = estimator_no_penalty.estimate(&program);
        let cost_high_penalty = estimator_high_penalty.estimate(&program);

        // 高いペナルティの方がコストが高い
        assert!(
            cost_high_penalty > cost_no_penalty,
            "Expected high penalty cost ({}) > no penalty cost ({})",
            cost_high_penalty,
            cost_no_penalty
        );
    }
}
