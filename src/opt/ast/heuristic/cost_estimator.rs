use crate::ast::{AstNode, ConstLiteral};
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
    fn const_literal_to_f32(c: &ConstLiteral) -> f32 {
        match *c {
            ConstLiteral::F32(v) => v,
            ConstLiteral::Isize(i) => i as f32,
            ConstLiteral::Usize(u) => u as f32,
            ConstLiteral::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
        }
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
                unroll,
                ..
            } => {
                let block_cost = self.estimate_cost_children(start);
                let mut overhead_cost = self.estimate_cost_children(body)
                    + self.estimate_cost_children(step)
                    + self.estimate_cost_children(max)
                    + 1e-8; // loop overhead
                let est_step = match step.as_ref() {
                    AstNode::Const(cl) => Self::const_literal_to_f32(cl),
                    _ => 1.0f32,
                };
                let est_max = match max.as_ref() {
                    AstNode::Const(cl) => Self::const_literal_to_f32(cl),
                    _ => 100f32,
                };
                let est_start = match start.as_ref() {
                    AstNode::Const(cl) => Self::const_literal_to_f32(cl),
                    _ => 0f32,
                };

                if let Some(factor) = unroll {
                    if *factor != 0 && *factor != 1 {
                        overhead_cost /= *factor as f32;
                    }
                }
                (block_cost + overhead_cost) * (est_max - est_start) / f32::max(est_step, 1.0)
                    + 1e-8
            }
            AstNode::Block { statements, .. } => {
                // Sum costs but cap individual statement costs to prevent explosion
                let mut total: f32 = 0.0;
                for stmt in statements {
                    let stmt_cost = self.estimate_cost(stmt);
                    total += stmt_cost
                }
                total
            }
            AstNode::Const(_) => 1e-9,
            AstNode::Var(_) => 1e-8,
            AstNode::Add(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Mul(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Rem(_, _) => 1e-8 + self.estimate_cost_children(ast),
            AstNode::Neg(_) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Recip(_) => 1e-8 + self.estimate_cost_children(ast),
            AstNode::Sin(_) => 2e-8 + self.estimate_cost_children(ast),
            AstNode::Sqrt(_) => 2e-8 + self.estimate_cost_children(ast),
            AstNode::Log2(_) => 2e-8 + self.estimate_cost_children(ast),
            AstNode::Exp2(_) => 2e-8 + self.estimate_cost_children(ast),
            AstNode::Max(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Store { .. } => 2e-9 + self.estimate_cost_children(ast),
            AstNode::Load { .. } => 5e-9 + self.estimate_cost_children(ast),
            _ => 1e-9 + self.estimate_cost_children(ast),
        }
    }
}

impl OperationCostEstimator {
    fn estimate_cost_children(&self, ast: &AstNode) -> f32 {
        ast.children()
            .iter()
            .map(|child| self.estimate_cost(child))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ConstLiteral, RangeBuilder, Scope, VariableDecl};

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
        let simple_loop = RangeBuilder::new(
            "i".to_string(),
            AstNode::Const(ConstLiteral::Isize(64)),
            simple_body.clone(),
        )
        .build();

        // Tiled loop structure (tile_size=16):
        // main_max = 64 - 64 % 16 = 64
        // for(i_tile_start=0; i_tile_start<64; i_tile_start+=16) {
        //     for(i=i_tile_start; i<i_tile_start+16; i++) { body }
        // }
        let inner_tiled_loop = RangeBuilder::new(
            "i".to_string(),
            AstNode::Add(
                Box::new(AstNode::Var("i_tile_start".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(16))),
            ),
            simple_body.clone(),
        )
        .start(AstNode::Var("i_tile_start".to_string()))
        .build();

        let main_tiled_loop = RangeBuilder::new(
            "i_tile_start".to_string(),
            AstNode::Var("i_main_max".to_string()),
            inner_tiled_loop,
        )
        .step(AstNode::Const(ConstLiteral::Isize(16)))
        .build();

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
