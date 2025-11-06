use crate::ast::AstNode;
use std::collections::HashMap;

/// ASTの実行コストを推定するトレイト
pub trait CostEstimator {
    /// ASTノードのコストを推定
    fn estimate(&self, ast: &AstNode) -> f64;
}

/// ASTノードのタイプ（コスト設定用）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Const,
    Var,
    Add,
    Mul,
    Max,
    Rem,
    Idiv,
    Recip,
    Sqrt,
    Log2,
    Exp2,
    Sin,
    Cast,
    Load,
    Store,
    Assign,
    Barrier,
    Block,
    Range,
    Call,
    Return,
}

impl NodeType {
    /// ASTノードからノードタイプを取得
    fn from_ast(ast: &AstNode) -> Self {
        match ast {
            AstNode::Const(_) => NodeType::Const,
            AstNode::Var(_) => NodeType::Var,
            AstNode::Add(_, _) => NodeType::Add,
            AstNode::Mul(_, _) => NodeType::Mul,
            AstNode::Max(_, _) => NodeType::Max,
            AstNode::Rem(_, _) => NodeType::Rem,
            AstNode::Idiv(_, _) => NodeType::Idiv,
            AstNode::Recip(_) => NodeType::Recip,
            AstNode::Sqrt(_) => NodeType::Sqrt,
            AstNode::Log2(_) => NodeType::Log2,
            AstNode::Exp2(_) => NodeType::Exp2,
            AstNode::Sin(_) => NodeType::Sin,
            AstNode::Cast(_, _) => NodeType::Cast,
            AstNode::Load { .. } => NodeType::Load,
            AstNode::Store { .. } => NodeType::Store,
            AstNode::Assign { .. } => NodeType::Assign,
            AstNode::Barrier => NodeType::Barrier,
            AstNode::Block { .. } => NodeType::Block,
            AstNode::Range { .. } => NodeType::Range,
            AstNode::Call { .. } => NodeType::Call,
            AstNode::Return { .. } => NodeType::Return,
            AstNode::Wildcard(_) => NodeType::Const, // ワイルドカードはコスト0として扱う
        }
    }

    /// デフォルトのコストを取得
    fn default_cost(&self) -> f64 {
        match self {
            NodeType::Const => 0.0,
            NodeType::Var => 0.0,
            NodeType::Add => 1.0,
            NodeType::Mul => 1.0,
            NodeType::Max => 1.0,
            NodeType::Rem => 2.0,
            NodeType::Idiv => 2.0,
            NodeType::Recip => 3.0,
            NodeType::Sqrt => 3.0,
            NodeType::Log2 => 4.0,
            NodeType::Exp2 => 4.0,
            NodeType::Sin => 5.0,
            NodeType::Cast => 1.0,
            NodeType::Load => 5.0,
            NodeType::Store => 5.0,
            NodeType::Assign => 0.5,
            NodeType::Barrier => 10.0,
            NodeType::Block => 0.0,
            NodeType::Range => 0.0,
            NodeType::Call => 5.0,
            NodeType::Return => 1.0,
        }
    }
}

/// 簡単なコスト推定器（ノード数ベース）
pub struct SimpleCostEstimator {
    /// ノードタイプごとのカスタムコスト重み
    custom_costs: HashMap<NodeType, f64>,
}

impl SimpleCostEstimator {
    /// 新しいコスト推定器を作成
    pub fn new() -> Self {
        Self {
            custom_costs: HashMap::new(),
        }
    }

    /// カスタムのノードコストを設定
    pub fn with_cost(mut self, node_type: NodeType, cost: f64) -> Self {
        self.custom_costs.insert(node_type, cost);
        self
    }

    /// ノードのベースコストを取得（カスタムコストがあればそれを、なければデフォルトを使用）
    fn base_cost(&self, ast: &AstNode) -> f64 {
        let node_type = NodeType::from_ast(ast);
        self.custom_costs
            .get(&node_type)
            .copied()
            .unwrap_or_else(|| node_type.default_cost())
    }
}

impl Default for SimpleCostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostEstimator for SimpleCostEstimator {
    fn estimate(&self, ast: &AstNode) -> f64 {
        let base_cost = self.base_cost(ast);

        // 子ノードのコストを再帰的に計算
        let children_cost: f64 = match ast {
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
                args.iter().map(|a| self.estimate(a)).sum::<f64>() + 5.0
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
