//! ルールベースの最適化器

use crate::ast::AstNode;
use crate::ast::pat::{AstRewriteRule, AstRewriter};
use crate::opt::ast::AstOptimizer;
use log::info;
use std::rc::Rc;

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

impl AstOptimizer for RuleBaseOptimizer {
    fn optimize(&mut self, ast: AstNode) -> AstNode {
        info!("AST rule-based optimization started");
        let result = self.rewriter.apply(ast);
        info!("AST rule-based optimization complete");
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
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::I64(0))))
        } => {
            a
        });

        let mut optimizer = RuleBaseOptimizer::new(vec![rule]);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::I64(42))),
            Box::new(AstNode::Const(Literal::I64(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::I64(42)));
    }
}
