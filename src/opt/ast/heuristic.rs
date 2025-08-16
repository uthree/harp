use crate::ast::AstNode;

pub trait CostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32;
}
pub trait Suggester {}
