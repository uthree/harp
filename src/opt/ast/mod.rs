use crate::ast::{AstNode, AstOp, pattern::AstRewriter};

pub mod heuristic;
pub mod rule;

pub trait AstOptimizer {
    fn optimize(&mut self, ast: &AstNode) -> AstNode;
}

pub struct CombinedAstOptimizer {
    optimizers: Vec<Box<dyn AstOptimizer>>,
}

impl AstOptimizer for CombinedAstOptimizer {
    fn optimize(&mut self, ast: &AstNode) -> AstNode {
        let mut ast = ast.clone();
        for opt in self.optimizers.iter_mut() {
            ast = opt.optimize(&ast);
        }
        ast
    }
}

impl CombinedAstOptimizer {
    pub fn new(optimizers: Vec<Box<dyn AstOptimizer>>) -> Self {
        CombinedAstOptimizer { optimizers }
    }
}

pub struct RulebasedAstOptimizer {
    rewriter: AstRewriter,
}

impl RulebasedAstOptimizer {
    pub fn new(rewriter: AstRewriter) -> Self {
        RulebasedAstOptimizer { rewriter: rewriter }
    }
}

impl AstOptimizer for RulebasedAstOptimizer {
    fn optimize(&mut self, ast: &AstNode) -> AstNode {
        self.rewriter.apply(ast)
    }
}
