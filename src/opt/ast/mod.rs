use crate::ast::AstNode;

pub mod algebraic;
pub mod heuristic;

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
