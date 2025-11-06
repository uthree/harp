use crate::ast::AstNode;
use crate::ast::pat::{AstRewriteRule, AstRewriter};
use log::debug;
use std::rc::Rc;

/// ASTを最適化するトレイト
pub trait Optimizer {
    /// ASTを最適化して返す
    fn optimize(&self, ast: AstNode) -> AstNode;
}

/// ルールベースの最適化器
pub struct RuleBaseOptimizer {
    rewriter: AstRewriter,
}

impl RuleBaseOptimizer {
    /// 新しい最適化器を作成
    pub fn new(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self {
            rewriter: AstRewriter::new(rules),
        }
    }

    /// 最大反復回数を設定
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.rewriter = self.rewriter.with_max_iterations(max);
        self
    }
}

impl Optimizer for RuleBaseOptimizer {
    fn optimize(&self, ast: AstNode) -> AstNode {
        debug!("RuleBaseOptimizer: Starting optimization");
        let result = self.rewriter.apply(ast);
        debug!("RuleBaseOptimizer: Optimization complete");
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;
    use crate::astpat;

    #[test]
    fn test_rule_base_optimizer() {
        // Add(a, 0) -> a というルール
        let rule = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Isize(0))))
        } => {
            a
        });

        let optimizer = RuleBaseOptimizer::new(vec![rule]);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(42))),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));
    }
}
