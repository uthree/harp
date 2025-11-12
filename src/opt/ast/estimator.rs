use crate::ast::AstNode;
const OVERHEAD_PER_LOOP: f32 = 1e-8;
/// ASTの実行コストを推定するトレイト
pub trait CostEstimator {
    /// ASTノードのコストを推定
    fn estimate(&self, ast: &AstNode) -> f32;
}

/// 簡単なコスト推定器（ノード数ベース）
pub struct SimpleCostEstimator;

impl SimpleCostEstimator {
    /// 新しいコスト推定器を作成
    pub fn new() -> Self {
        Self
    }

    /// ノードのベースコストを取得
    fn base_cost(&self, ast: &AstNode) -> f32 {
        let cost = match ast {
            AstNode::Const(_) | AstNode::Wildcard(_) => 0.01,
            AstNode::Var(_) => 1.2,
            AstNode::Add(_, _) => 1.0,
            AstNode::Mul(_, _) => 4.0,
            AstNode::Max(_, _) => 3.0,
            AstNode::Rem(_, _) => 5.0,
            AstNode::Idiv(_, _) => 2.0,
            AstNode::Recip(_) => 10.0,
            AstNode::Sqrt(_) => 20.0,
            AstNode::Log2(_) => 20.0,
            AstNode::Exp2(_) => 20.0,
            AstNode::Sin(_) => 20.0,
            AstNode::Cast(_, _) => 4.0,
            // Bitwise operations - ビット演算（シフトは乗算より低コスト）
            AstNode::BitwiseAnd(_, _) => 0.5,
            AstNode::BitwiseOr(_, _) => 0.5,
            AstNode::BitwiseXor(_, _) => 0.5,
            AstNode::BitwiseNot(_) => 0.5,
            AstNode::LeftShift(_, _) => 0.8, // 乗算(4.0)より低コスト
            AstNode::RightShift(_, _) => 0.8, // 乗算(4.0)より低コスト

            AstNode::Load { .. } => 2.0,
            AstNode::Store { .. } => 2.0,
            AstNode::Assign { .. } => 1.2,
            AstNode::Barrier => 0.1,
            AstNode::Block { .. } => 0.01,
            AstNode::Range { .. } => 0.01,
            AstNode::Call { .. } => 1.0,
            AstNode::Return { .. } => 0.01,
            AstNode::Function { .. } => 0.01,
            AstNode::Program { .. } => 0.01, // プログラム構造自体にはコストがない
        };
        cost * 1e-9
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

        // 子ノードのコストを再帰的に計算
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
            | AstNode::RightShift(l, r) => self.estimate(l) + self.estimate(r),
            AstNode::Recip(n)
            | AstNode::Sqrt(n)
            | AstNode::Log2(n)
            | AstNode::Exp2(n)
            | AstNode::Sin(n)
            | AstNode::BitwiseNot(n) => self.estimate(n),
            AstNode::Cast(n, _) => self.estimate(n),
            AstNode::Load { ptr, offset, .. } => self.estimate(ptr) + self.estimate(offset),
            AstNode::Store { ptr, offset, value } => {
                self.estimate(ptr) + self.estimate(offset) + self.estimate(value)
            }
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
                self.estimate(start)
                    + (self.estimate(body)
                        + OVERHEAD_PER_LOOP
                        + self.estimate(step)
                        + self.estimate(stop))
                        * loop_count
            }
            AstNode::Block { statements, .. } => statements.iter().map(|s| self.estimate(s)).sum(),
            AstNode::Call { args, .. } => {
                // 関数呼び出しは引数の評価コスト + 呼び出しコスト
                args.iter().map(|a| self.estimate(a)).sum::<f32>()
            }
            AstNode::Return { value } => self.estimate(value),
            AstNode::Function { body, params, .. } => {
                // 関数本体のコスト + パラメータの初期値のコスト
                let body_cost = self.estimate(body);
                let params_cost: f32 = params
                    .iter()
                    .filter_map(|p| p.initial_value.as_ref())
                    .map(|init| self.estimate(init))
                    .sum();
                body_cost + params_cost
            }
            AstNode::Program { functions, .. } => {
                // すべての関数のコストの合計
                functions.iter().map(|f| self.estimate(f)).sum()
            }
            _ => 0.01,
        };

        base_cost + children_cost
    }
}
