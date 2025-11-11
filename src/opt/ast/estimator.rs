use crate::ast::{AstNode, Literal};

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
            AstNode::Const(_) | AstNode::Wildcard(_) => 0.0,
            AstNode::Var(_) => 0.0,
            AstNode::Add(_, _) => 1.0,
            AstNode::Mul(_, _) => 2.0,
            AstNode::Max(_, _) => 1.0,
            AstNode::Rem(_, _) => 5.0,
            AstNode::Idiv(_, _) => 2.0,
            AstNode::Recip(_) => 10.0,
            AstNode::Sqrt(_) => 15.0,
            AstNode::Log2(_) => 15.0,
            AstNode::Exp2(_) => 15.0,
            AstNode::Sin(_) => 15.0,
            AstNode::Cast(_, _) => 4.0,
            AstNode::Load { .. } => 10.0,
            AstNode::Store { .. } => 10.0,
            AstNode::Assign { .. } => 10.0,
            AstNode::Barrier => 10.0,
            AstNode::Block { .. } => 10.0,
            AstNode::Range { .. } => 10.0,
            AstNode::Call { .. } => 20.0,
            AstNode::Return { .. } => 0.0,
            AstNode::Function { .. } => 0.0, // 関数定義自体にはコストがない
            AstNode::Program { .. } => 0.0,  // プログラム構造自体にはコストがない
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
            | AstNode::Idiv(l, r) => self.estimate(l) + self.estimate(r),
            AstNode::Recip(n)
            | AstNode::Sqrt(n)
            | AstNode::Log2(n)
            | AstNode::Exp2(n)
            | AstNode::Sin(n) => self.estimate(n),
            AstNode::Cast(n, _) => self.estimate(n),
            AstNode::Load {
                ptr, offset, count, ..
            } => self.estimate(ptr) + self.estimate(offset) + 1e-9 * (*count as f32),
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
                        AstNode::Const(Literal::Isize(start_val)),
                        AstNode::Const(Literal::Isize(stop_val)),
                        AstNode::Const(Literal::Isize(step_val)),
                    ) if *step_val > 0 => {
                        // 正の方向のループ
                        let iterations = (stop_val - start_val + step_val - 1) / step_val;
                        iterations.max(0) as f32
                    }
                    (
                        AstNode::Const(Literal::Isize(start_val)),
                        AstNode::Const(Literal::Isize(stop_val)),
                        AstNode::Const(Literal::Isize(step_val)),
                    ) if *step_val < 0 => {
                        // 負の方向のループ
                        let iterations = (start_val - stop_val - step_val - 1) / (-step_val);
                        iterations.max(0) as f32
                    }
                    _ => {
                        // ループ回数が不明な場合は100回と推定
                        100.0
                    }
                };

                self.estimate(start)
                    + self.estimate(step)
                    + self.estimate(stop)
                    + self.estimate(body) * loop_count
            }
            AstNode::Block { statements, .. } => statements.iter().map(|s| self.estimate(s)).sum(),
            AstNode::Call { args, .. } => {
                // 関数呼び出しは引数の評価コスト + 呼び出しコスト
                args.iter().map(|a| self.estimate(a)).sum::<f32>() + 5.0
            }
            AstNode::Return { value } => self.estimate(value) + 1.0,
            _ => 0.0,
        };

        base_cost + children_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_simple_cost_estimator() {
        let estimator = SimpleCostEstimator::new();

        // 定数のコスト
        let const_node = AstNode::Const(Literal::Isize(42));
        assert_eq!(estimator.estimate(&const_node), 0.0);

        // 加算のコスト (base_cost: 1.0 * 1e-9)
        let add_node = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(1))),
            Box::new(AstNode::Const(Literal::Isize(2))),
        );
        assert_eq!(estimator.estimate(&add_node), 1.0 * 1e-9);

        // 平方根のコスト (base_cost: 15.0 * 1e-9)
        let sqrt_node = AstNode::Sqrt(Box::new(AstNode::Const(Literal::F32(4.0))));
        assert_eq!(estimator.estimate(&sqrt_node), 15.0 * 1e-9);

        // 複合演算のコスト: (a + b) * c
        let complex_node = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::Isize(1))),
                Box::new(AstNode::Const(Literal::Isize(2))),
            )),
            Box::new(AstNode::Const(Literal::Isize(3))),
        );
        // Add: 1.0 * 1e-9, Mul: 2.0 * 1e-9 + Add.cost = 3.0 * 1e-9
        assert_eq!(estimator.estimate(&complex_node), 3.0 * 1e-9);
    }

    #[test]
    fn test_cost_comparison() {
        let estimator = SimpleCostEstimator::new();

        // 2つの等価な式のコストを比較
        // (a + 0) * 1
        let expr1 = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(Literal::Isize(0))),
            )),
            Box::new(AstNode::Const(Literal::Isize(1))),
        );

        // a
        let expr2 = AstNode::Var("a".to_string());

        let cost1 = estimator.estimate(&expr1);
        let cost2 = estimator.estimate(&expr2);

        // expr1の方がコストが高いはず
        assert!(cost1 > cost2);
    }

    #[test]
    fn test_range_cost_with_known_iterations() {
        let estimator = SimpleCostEstimator::new();

        // ループ回数が明確な場合（0から10まで、ステップ1）
        let range_10 = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            stop: Box::new(AstNode::Const(Literal::Isize(10))),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            body: Box::new(AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("i".to_string())),
            )),
        };

        // bodyのコストを計算
        let body_cost = estimator.estimate(&AstNode::Add(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("i".to_string())),
        ));

        // ループ回数は10回なので、children_costは 0 + 0 + 0 + body_cost * 10
        let cost_10 = estimator.estimate(&range_10);
        let expected_cost_10 = 10.0 * body_cost + 10.0 * 1e-9; // children_cost + base_cost
        assert!((cost_10 - expected_cost_10).abs() < 1e-12);

        // ループ回数が100回の場合
        let range_100 = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            stop: Box::new(AstNode::Const(Literal::Isize(100))),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            body: Box::new(AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("i".to_string())),
            )),
        };

        let cost_100 = estimator.estimate(&range_100);
        let expected_cost_100 = 100.0 * body_cost + 10.0 * 1e-9;
        assert!((cost_100 - expected_cost_100).abs() < 1e-12);

        // ループ回数が不明な場合（変数を使用）
        let range_unknown = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            stop: Box::new(AstNode::Var("n".to_string())),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            body: Box::new(AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("i".to_string())),
            )),
        };

        // ループ回数が不明なので100回と推定され、children_costは 0 + 0 + 0 + body_cost * 100
        let cost_unknown = estimator.estimate(&range_unknown);
        let expected_cost_unknown = 100.0 * body_cost + 10.0 * 1e-9;
        assert!((cost_unknown - expected_cost_unknown).abs() < 1e-12);

        // 重要な比較：明確な回数のループと不明な回数のループ
        // ループ10回の方がループ100回より大幅にコストが低いはず
        assert!(cost_10 < cost_100);
        // ループ回数不明（100回推定）とループ100回は同じコストのはず
        assert!((cost_100 - cost_unknown).abs() < 1e-12);
    }
}
