use crate::ast::AstNode;
use crate::opt::ast::heuristic::CostEstimator;

/// A simple cost estimator that counts the number of nodes in the AST.
#[derive(Clone, Copy)]
pub struct NodeCountCostEstimator;

impl CostEstimator for NodeCountCostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32 {
        1.0 + ast
            .children()
            .iter()
            .map(|child| self.estimate_cost(child))
            .sum::<f32>()
    }
}

/// A cost estimator that assigns different costs to different operations.
#[derive(Clone, Copy)]
pub struct OperationCostEstimator;

impl CostEstimator for OperationCostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32 {
        let base_cost = match ast {
            AstNode::Const(_) => 0.0,
            AstNode::Var(_) => 1.0,
            AstNode::Add(_, _) => 1.0,
            AstNode::Mul(_, _) => 2.0,
            AstNode::Div(_, _) => 4.0,
            AstNode::Rem(_, _) => 4.0,
            AstNode::Neg(_) => 1.0,
            AstNode::Recip(_) => 4.0,
            AstNode::Sin(_) => 10.0,
            AstNode::Sqrt(_) => 5.0,
            AstNode::Log2(_) => 8.0,
            AstNode::Exp2(_) => 8.0,
            AstNode::Max(_, _) => 1.0,
            AstNode::Range { .. } => 5.0,
            AstNode::Store { .. } => 3.0,
            AstNode::Deref(_) => 2.0,
            _ => 1.0,
        };

        base_cost
            + ast
                .children()
                .iter()
                .map(|child| self.estimate_cost(child))
                .sum::<f32>()
    }
}
