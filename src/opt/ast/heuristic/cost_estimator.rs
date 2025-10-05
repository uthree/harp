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
/// Also recognizes tiled loops and applies a cost reduction for better cache locality.
#[derive(Clone, Copy)]
pub struct OperationCostEstimator;

impl OperationCostEstimator {
    /// Check if a Range node is a tiled loop (outer loop of a tiling pattern)
    fn is_tiled_loop(node: &AstNode) -> bool {
        if let AstNode::Range {
            counter_name,
            step,
            body,
            ..
        } = node
        {
            // Check if this is a tile_start loop with step > 1
            if counter_name.contains("_tile_start") {
                if let AstNode::Const(crate::ast::ConstLiteral::Isize(step_val)) = **step {
                    if step_val > 1 {
                        // Check if body contains a nested loop
                        if let AstNode::Block { statements, .. } = &**body {
                            return statements.iter().any(|stmt| matches!(stmt, AstNode::Range { .. }));
                        }
                        return matches!(**body, AstNode::Range { .. });
                    }
                }
            }
        }
        false
    }
}

impl CostEstimator for OperationCostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32 {
        let mut base_cost = match ast {
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

        // Apply cost reduction for tiled loops (better cache locality)
        if Self::is_tiled_loop(ast) {
            // Reduce cost by 50% to strongly favor tiled loops
            // This accounts for improved cache locality and reduced memory bandwidth
            base_cost *= 0.5;
        }

        // Also check if this is a Block containing a tiled loop pattern
        if let AstNode::Block { statements, .. } = ast {
            // Check if this block contains both a tiled loop and a remainder loop
            let has_tiled = statements.iter().any(|s| Self::is_tiled_loop(s));
            let has_remainder = statements.iter().filter(|s| matches!(s, AstNode::Range { .. })).count() > 1;
            if has_tiled && has_remainder {
                // This is likely a complete tiling pattern, apply additional discount
                base_cost *= 0.8;
            }
        }

        base_cost
            + ast
                .children()
                .iter()
                .map(|child| self.estimate_cost(child))
                .sum::<f32>()
    }
}
