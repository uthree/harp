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
/// Uses logarithmic scale to prevent cost explosion in nested loops.
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
                            return statements
                                .iter()
                                .any(|stmt| matches!(stmt, AstNode::Range { .. }));
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
        match ast {
            AstNode::Range {
                start,
                max,
                step,
                body,
                ..
            } => {
                // Calculate loop iterations: (max - start) / step
                let iterations = self.estimate_iterations(start, max, step);
                let body_cost = self.estimate_cost(body);
                let overhead: f32 = 2.0; // Loop overhead (init, check, increment)

                // Use logarithmic scale: log(iterations) + body_cost instead of iterations * body_cost
                // This prevents cost explosion in nested loops
                let log_iterations = (iterations.max(1.0)).log2();
                let mut total_cost = overhead.max(log_iterations + body_cost);

                // Apply cost reduction for tiled loops (better cache locality)
                if Self::is_tiled_loop(ast) {
                    // Reduce cost by 30% to favor tiled loops
                    // This accounts for improved cache locality
                    total_cost -= 3.0; // Subtract instead of multiply in log scale
                }

                total_cost
            }
            AstNode::Block { statements, .. } => {
                // Sum costs but cap individual statement costs to prevent explosion
                let mut total: f32 = 0.0;
                for stmt in statements {
                    let stmt_cost = self.estimate_cost(stmt);
                    // In log scale, adding costs means sequential execution
                    total += stmt_cost.min(20.0); // Cap to prevent explosion
                }

                // Apply discount for complete tiling pattern
                let has_tiled = statements.iter().any(Self::is_tiled_loop);
                let has_remainder = statements
                    .iter()
                    .filter(|s| matches!(s, AstNode::Range { .. }))
                    .count()
                    > 1;
                if has_tiled && has_remainder {
                    // Additional discount for complete tiling pattern
                    total -= 2.0; // Subtract in log scale
                } else if has_tiled {
                    // Even if there's no remainder, give a discount for tiling
                    total -= 5.0; // Large discount to favor tiling
                }

                total
            }
            AstNode::Const(_) => 0.0,
            AstNode::Var(_) => 1.0,
            AstNode::Add(_, _) => 1.0 + self.estimate_cost_children_max(ast),
            AstNode::Mul(_, _) => 2.0 + self.estimate_cost_children_max(ast),
            AstNode::Div(_, _) => 4.0 + self.estimate_cost_children_max(ast),
            AstNode::Rem(_, _) => 4.0 + self.estimate_cost_children_max(ast),
            AstNode::Neg(_) => 1.0 + self.estimate_cost_children_max(ast),
            AstNode::Recip(_) => 4.0 + self.estimate_cost_children_max(ast),
            AstNode::Sin(_) => 10.0 + self.estimate_cost_children_max(ast),
            AstNode::Sqrt(_) => 5.0 + self.estimate_cost_children_max(ast),
            AstNode::Log2(_) => 8.0 + self.estimate_cost_children_max(ast),
            AstNode::Exp2(_) => 8.0 + self.estimate_cost_children_max(ast),
            AstNode::Max(_, _) => 1.0 + self.estimate_cost_children_max(ast),
            AstNode::Store { .. } => 3.0 + self.estimate_cost_children_max(ast),
            AstNode::Deref(_) => 2.0 + self.estimate_cost_children_max(ast),
            _ => 1.0 + self.estimate_cost_children_max(ast),
        }
    }
}

impl OperationCostEstimator {
    fn estimate_cost_children_max(&self, ast: &AstNode) -> f32 {
        ast.children()
            .iter()
            .map(|child| self.estimate_cost(child))
            .fold(0.0, f32::max)
    }

    fn estimate_iterations(&self, start: &AstNode, max: &AstNode, step: &AstNode) -> f32 {
        // Try to extract constant values
        let start_val = if let AstNode::Const(crate::ast::ConstLiteral::Isize(v)) = start {
            *v as f32
        } else {
            0.0 // Unknown, assume 0
        };

        let max_val = if let AstNode::Const(crate::ast::ConstLiteral::Isize(v)) = max {
            *v as f32
        } else {
            100.0 // Unknown, use conservative estimate
        };

        let step_val = if let AstNode::Const(crate::ast::ConstLiteral::Isize(v)) = step {
            (*v as f32).max(1.0) // Avoid division by zero
        } else {
            1.0 // Unknown, assume 1
        };

        ((max_val - start_val) / step_val).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ConstLiteral, Scope, VariableDecl};

    #[test]
    fn test_tiled_loop_cost_is_lower() {
        let estimator = OperationCostEstimator;

        // Simple loop: for(i=0; i<64; i++) { body }
        let simple_body = AstNode::Assign(
            "x".to_string(),
            Box::new(AstNode::Add(
                Box::new(AstNode::Var("x".to_string())),
                Box::new(AstNode::Var("i".to_string())),
            )),
        );
        let simple_loop = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(64))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(simple_body.clone()),
        };

        // Tiled loop structure (tile_size=16):
        // main_max = 64 - 64 % 16 = 64
        // for(i_tile_start=0; i_tile_start<64; i_tile_start+=16) {
        //     for(i=i_tile_start; i<i_tile_start+16; i++) { body }
        // }
        let inner_tiled_loop = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Var("i_tile_start".to_string())),
            max: Box::new(AstNode::Add(
                Box::new(AstNode::Var("i_tile_start".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(16))),
            )),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(simple_body.clone()),
        };

        let main_tiled_loop = AstNode::Range {
            counter_name: "i_tile_start".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Var("i_main_max".to_string())),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(16))),
            body: Box::new(inner_tiled_loop),
        };

        let tiled_block = AstNode::Block {
            scope: Scope {
                declarations: vec![VariableDecl {
                    name: "i_main_max".to_string(),
                    dtype: crate::ast::DType::Isize,
                    constant: false,
                    size_expr: None,
                }],
            },
            statements: vec![
                AstNode::Assign(
                    "i_main_max".to_string(),
                    Box::new(AstNode::Const(ConstLiteral::Isize(64))),
                ),
                main_tiled_loop,
            ],
        };

        let simple_cost = estimator.estimate_cost(&simple_loop);
        let tiled_cost = estimator.estimate_cost(&tiled_block);

        // Tiled loop should have lower cost due to the discount
        assert!(
            tiled_cost < simple_cost,
            "Tiled loop cost ({}) should be lower than simple loop cost ({})",
            tiled_cost,
            simple_cost
        );
    }
}
