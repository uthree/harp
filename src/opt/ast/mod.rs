use crate::ast::AstNode;

pub mod heuristic;

pub trait AstOptimizer {
    fn new() -> Self;
    fn optimize(&mut self, ast: &AstNode) -> AstNode;
}

pub struct CombinedAstOptimizer<A: AstOptimizer, B: AstOptimizer> {
    a: A,
    b: B,
}

impl<A, B> AstOptimizer for CombinedAstOptimizer<A, B>
where
    A: AstOptimizer,
    B: AstOptimizer,
{
    fn new() -> Self {
        Self {
            a: A::new(),
            b: B::new(),
        }
    }
    fn optimize(&mut self, ast: &AstNode) -> AstNode {
        self.b.optimize(&self.a.optimize(ast))
    }
}
