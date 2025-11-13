use super::CostEstimator;
use crate::ast::AstNode;
use crate::opt::cost_utils::{log_sum_exp, log_sum_exp_iter};

// ループのオーバーヘッド（ループカウンタのインクリメント、比較、分岐）
const OVERHEAD_PER_LOOP: f32 = 2.0;

// メモリアクセスのコスト（L1キャッシュヒット想定）
const MEMORY_ACCESS_COST: f32 = 4.0;

// 関数呼び出しのオーバーヘッド（スタックフレームの設定、レジスタ退避など）
const FUNCTION_CALL_OVERHEAD: f32 = 10.0;

// 同期バリアのコスト（スレッド間の同期待ち）
const BARRIER_COST: f32 = 100.0;

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
pub struct SimpleCostEstimator;

impl SimpleCostEstimator {
    /// 新しいコスト推定器を作成
    pub fn new() -> Self {
        Self
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
            AstNode::Load { count, .. } => (MEMORY_ACCESS_COST * (*count as f32)).ln(),
            AstNode::Store { .. } => MEMORY_ACCESS_COST.ln(),

            // 型変換（整数↔浮動小数点など）
            AstNode::Cast(_, _) => 2.0_f32.ln(),

            // 同期バリア（スレッド間の同期待ち）
            AstNode::Barrier => BARRIER_COST.ln(),

            // その他（変数参照、定数など）
            _ => f32::NEG_INFINITY, // log(0) = -∞
        }
    }
}

impl Default for SimpleCostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostEstimator for SimpleCostEstimator {
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
                    OVERHEAD_PER_LOOP.ln(),
                ]);
                log_sum_exp(self.estimate(start), log_loop_count + per_iteration_cost)
            }
            AstNode::Block { statements, .. } => {
                log_sum_exp_iter(statements.iter().map(|s| self.estimate(s)))
            }
            AstNode::Call { args, .. } => {
                // 関数呼び出しは引数の評価コスト + 呼び出しオーバーヘッド
                let args_cost = log_sum_exp_iter(args.iter().map(|a| self.estimate(a)));
                log_sum_exp(args_cost, FUNCTION_CALL_OVERHEAD.ln())
            }
            AstNode::Return { value } => self.estimate(value),
            AstNode::Function { body, params, .. } => {
                // 関数本体のコスト + パラメータの初期値のコスト
                let body_cost = self.estimate(body);
                let params_cost = log_sum_exp_iter(
                    params
                        .iter()
                        .filter_map(|p| p.initial_value.as_ref())
                        .map(|init| self.estimate(init)),
                );
                log_sum_exp(body_cost, params_cost)
            }
            AstNode::Program { functions, .. } => {
                // すべての関数のコストの合計
                log_sum_exp_iter(functions.iter().map(|f| self.estimate(f)))
            }
            _ => f32::NEG_INFINITY, // log(0)
        };

        log_sum_exp(base_cost, children_cost)
    }
}
