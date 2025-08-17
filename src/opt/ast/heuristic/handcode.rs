use crate::ast::{AstNode, AstOp};
use crate::opt::ast::heuristic::CostEstimator;

#[derive(Clone, Copy)]
pub struct HandcodedCostEstimator;

impl CostEstimator for HandcodedCostEstimator {
    fn estimate_cost(&self, node: &AstNode) -> f32 {
        match &node.op {
            AstOp::Range { step, .. } => {
                if node.src.is_empty() {
                    return 0.0; // No loop bound or body
                }
                let max_node = &node.src[0];
                let body_nodes = &node.src[1..];

                let max_node_cost = self.estimate_cost(max_node);
                let body_cost: f32 = body_nodes.iter().map(|n| self.estimate_cost(n)).sum();
                let body_cost: f32 = body_cost + 5e-8f32; // The cost of loop counters required for iteration, conditional branches, and program counter movements

                let iterations = if let AstOp::Const(c) = &max_node.op {
                    if let Some(val) = c.as_isize() {
                        // Assuming start is 0 and step is a positive integer
                        (val as f32 / *step as f32).ceil().max(0.0)
                    } else {
                        100.0 // Non-integer constant for loop bound, use heuristic
                    }
                } else {
                    100.0 // Dynamic loop bound, use a heuristic multiplier
                };

                max_node_cost + (body_cost * iterations)
            }
            _ => {
                let op_cost = match &node.op {
                    AstOp::Div => 10.0,
                    AstOp::Store => 8.0,
                    AstOp::Assign => 5.0,
                    AstOp::Declare { .. } => 0.0,
                    AstOp::Func { .. } => 0.0,
                    AstOp::Var(_) => 2.0,
                    AstOp::Call(_) => 5.0,
                    _ => 1.0,
                } * 1e-9f32;
                let children_cost: f32 =
                    node.src.iter().map(|child| self.estimate_cost(child)).sum();
                op_cost + children_cost
            }
        }
    }
}
