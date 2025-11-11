use crate::ast::AstNode;

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
        match ast {
            AstNode::Const(_) | AstNode::Wildcard(_) => 0.0,
            AstNode::Var(_) => 0.0,
            AstNode::Add(_, _) => 1.0,
            AstNode::Mul(_, _) => 1.0,
            AstNode::Max(_, _) => 1.0,
            AstNode::Rem(_, _) => 2.0,
            AstNode::Idiv(_, _) => 2.0,
            AstNode::Recip(_) => 3.0,
            AstNode::Sqrt(_) => 3.0,
            AstNode::Log2(_) => 4.0,
            AstNode::Exp2(_) => 4.0,
            AstNode::Sin(_) => 5.0,
            AstNode::Cast(_, _) => 1.0,
            AstNode::Load { .. } => 5.0,
            AstNode::Store { .. } => 5.0,
            AstNode::Assign { .. } => 0.5,
            AstNode::Barrier => 10.0,
            AstNode::Block { .. } => 0.0,
            AstNode::Range { .. } => 0.0,
            AstNode::Call { .. } => 5.0,
            AstNode::Return { .. } => 1.0,
            AstNode::Function { .. } => 0.0, // 関数定義自体にはコストがない
            AstNode::Program { .. } => 0.0,  // プログラム構造自体にはコストがない
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
                // ループは推定回数を10として計算
                self.estimate(start)
                    + self.estimate(step)
                    + self.estimate(stop)
                    + self.estimate(body) * 10.0
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

        // 加算のコスト
        let add_node = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(1))),
            Box::new(AstNode::Const(Literal::Isize(2))),
        );
        assert_eq!(estimator.estimate(&add_node), 1.0);

        // 平方根のコスト
        let sqrt_node = AstNode::Sqrt(Box::new(AstNode::Const(Literal::F32(4.0))));
        assert_eq!(estimator.estimate(&sqrt_node), 3.0);

        // 複合演算のコスト: (a + b) * c
        let complex_node = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::Isize(1))),
                Box::new(AstNode::Const(Literal::Isize(2))),
            )),
            Box::new(AstNode::Const(Literal::Isize(3))),
        );
        assert_eq!(estimator.estimate(&complex_node), 2.0); // 1.0 (Add) + 1.0 (Mul)
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
}
