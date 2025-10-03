use crate::ast::{pattern::AstRewriter, AstNode};
pub mod constant_folding;
pub mod heuristic;
pub mod simplify;
pub mod suggester;

pub trait AstOptimizer {
    fn optimize(&mut self, ast: &mut AstNode);
}

pub struct CombinedAstOptimizer {
    optimizers: Vec<Box<dyn AstOptimizer>>,
}

impl AstOptimizer for CombinedAstOptimizer {
    fn optimize(&mut self, ast: &mut AstNode) {
        for opt in self.optimizers.iter_mut() {
            opt.optimize(ast);
        }
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
    fn optimize(&mut self, ast: &mut AstNode) {
        loop {
            let before = ast.clone();
            self.rewriter.apply(ast);
            if *ast == before {
                break;
            }
        }
    }
}
