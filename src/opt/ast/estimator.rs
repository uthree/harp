use super::CostEstimator;
use crate::ast::AstNode;
use crate::opt::cost_utils::{log_sum_exp, log_sum_exp_iter};

// ループのオーバーヘッド（ループカウンタのインクリメント、比較、分岐）
const OVERHEAD_PER_LOOP: f32 = 2.0;

// メモリアクセスのコスト（L1キャッシュヒット想定）
const MEMORY_ACCESS_COST: f32 = 4.0;

// 関数呼び出しのオーバーヘッド（スタックフレームの設定、レジスタ退避など）
const FUNCTION_CALL_OVERHEAD: f32 = 10.0;

// 関数定義のオーバーヘッド（プロローグ/エピローグ生成、シンボルテーブルエントリなど）
const FUNCTION_DEFINITION_OVERHEAD: f32 = 50.0;

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
            AstNode::Function { body, .. } => {
                // 関数本体のコスト + 関数定義オーバーヘッド
                // 関数定義自体にもコストがかかる（プロローグ/エピローグ、スタックフレーム管理など）
                log_sum_exp(self.estimate(body), FUNCTION_DEFINITION_OVERHEAD.ln())
            }
            AstNode::Program {
                functions,
                entry_point,
            } => {
                // エントリポイント以外の関数の数に基づいてペナルティを計算
                // 未使用の関数が多いほどコストが高くなる（インライン展開を促進）
                let non_entry_functions = functions
                    .iter()
                    .filter(|f| {
                        if let AstNode::Function { name: Some(n), .. } = f {
                            n != entry_point
                        } else {
                            true
                        }
                    })
                    .count();

                // すべての関数のコストの合計
                let functions_cost = log_sum_exp_iter(functions.iter().map(|f| self.estimate(f)));

                // 非エントリポイント関数の数に比例したペナルティを追加
                // 関数が多いほど、コードサイズが大きくなり、キャッシュ効率が悪化する
                if non_entry_functions > 0 {
                    let penalty = (non_entry_functions as f32 * FUNCTION_DEFINITION_OVERHEAD).ln();
                    log_sum_exp(functions_cost, penalty)
                } else {
                    functions_cost
                }
            }
            _ => f32::NEG_INFINITY, // log(0)
        };

        log_sum_exp(base_cost, children_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, FunctionKind, Literal, Mutability, Scope, VarDecl, VarKind};

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
            kind: FunctionKind::Normal,
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
            kind: FunctionKind::Normal,
        };

        let program_before = AstNode::Program {
            functions: vec![add_one_func, main_with_call],
            entry_point: "main".to_string(),
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
            kind: FunctionKind::Normal,
        };

        let program_after = AstNode::Program {
            functions: vec![main_inlined],
            entry_point: "main".to_string(),
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
                kind: FunctionKind::Normal,
            }],
            entry_point: "main".to_string(),
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
                    kind: FunctionKind::Normal,
                },
                AstNode::Function {
                    name: Some("main".to_string()),
                    params: vec![],
                    return_type: DType::Int,
                    body: Box::new(AstNode::Return {
                        value: Box::new(AstNode::Const(Literal::Int(1))),
                    }),
                    kind: FunctionKind::Normal,
                },
            ],
            entry_point: "main".to_string(),
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
            kind: FunctionKind::Normal,
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
            kind: FunctionKind::Normal,
        };

        let program_before = AstNode::Program {
            functions: vec![write_value_func, main_with_call],
            entry_point: "main".to_string(),
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
            kind: FunctionKind::Normal,
        };

        let program_after = AstNode::Program {
            functions: vec![main_inlined],
            entry_point: "main".to_string(),
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
}
