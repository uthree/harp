use crate::ast::{pattern::AstRewriter, AstNode};
pub mod constant_folding;
pub mod heuristic;
pub mod simplify;
pub mod suggester;

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

// AstRewriterに従って項書き換えを行うだけのOptimizer
pub struct RulebasedAstOptimizer {
    rewriter: AstRewriter,
}

impl RulebasedAstOptimizer {
    pub fn new(rewriter: AstRewriter) -> Self {
        RulebasedAstOptimizer { rewriter }
    }
}

impl AstOptimizer for RulebasedAstOptimizer {
    fn optimize(&mut self, ast: &AstNode) -> AstNode {
        let mut current_ast = ast.clone();
        loop {
            let next_ast = self.rewriter.apply(&current_ast);
            if next_ast == current_ast {
                return next_ast;
            }
            current_ast = next_ast;
        }
    }
}
