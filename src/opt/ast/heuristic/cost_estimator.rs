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

            // Comparison and conditional operations
            AstNode::LessThan(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Eq(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Select { .. } => 1e-9 + self.estimate_cost_children(ast),
            AstNode::If {
                condition,
                then_branch,
                else_branch,
            } => {
                // If statement cost = condition cost + average of branches
                // We use average because we don't know which branch will be taken at compile time
                let cond_cost = self.estimate_cost(condition);
                let then_cost = self.estimate_cost(then_branch);
                let else_cost = else_branch
                    .as_ref()
                    .map(|e| self.estimate_cost(e))
                    .unwrap_or(0.0);
                1e-9 + cond_cost + (then_cost + else_cost) * 0.5
            }

            // Bitwise operations
            AstNode::BitAnd(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::BitOr(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::BitXor(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Shl(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Shr(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::BitNot(_) => 1e-9 + self.estimate_cost_children(ast),

            // Type conversion
            AstNode::Cast { .. } => 1e-9 + self.estimate_cost_children(ast),

            // Random number generation (expensive)
            AstNode::Rand => 2e-8,

            // Barrier (synchronization overhead)
            AstNode::Barrier => 1e-7,

            // Function and kernel operations
            AstNode::CallFunction { args, .. } => {
                // Function call overhead + argument evaluation costs
                let args_cost: f32 = args.iter().map(|arg| self.estimate_cost(arg)).sum();
                1e-7 + args_cost
            }
            AstNode::Function { statements, .. } => {
                // Function definition itself has no runtime cost, only the statements inside
                statements.iter().map(|stmt| self.estimate_cost(stmt)).sum()
            }
            AstNode::Kernel { statements, .. } => {
                // Kernel definition has higher overhead due to launch costs
                let body_cost: f32 = statements.iter().map(|stmt| self.estimate_cost(stmt)).sum();
                1e-6 + body_cost // Kernel launch overhead
            }
            AstNode::CallKernel { args, .. } => {
                // Kernel call has significant overhead
                let args_cost: f32 = args.iter().map(|arg| self.estimate_cost(arg)).sum();
                1e-6 + args_cost // Kernel launch overhead
            }
            AstNode::Program { functions, .. } => {
                // Program cost is sum of all functions
                functions.iter().map(|func| self.estimate_cost(func)).sum()
            }

            // Memory management
            AstNode::Assign(_, _) => 1e-9 + self.estimate_cost_children(ast),
            AstNode::Drop(_) => 1e-9, // Negligible cost

            // Pattern matching (should not appear in runtime code)
            AstNode::Capture(_) => 0.0,
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

    #[test]
    fn test_if_statement_cost() {
        use crate::ast::helper::*;
        let estimator = OperationCostEstimator;

        // Simple if without else: if (i < n) x = 1;
        let if_stmt = if_then(
            AstNode::LessThan(
                Box::new(AstNode::Var("i".to_string())),
                Box::new(AstNode::Var("n".to_string())),
            ),
            AstNode::Assign("x".to_string(), Box::new(AstNode::from(1_isize))),
        );

        let cost = estimator.estimate_cost(&if_stmt);
        assert!(cost > 0.0, "If statement should have non-zero cost");

        // If with else should have higher cost than simple if
        let if_else_stmt = if_then_else(
            AstNode::LessThan(
                Box::new(AstNode::Var("i".to_string())),
                Box::new(AstNode::Var("n".to_string())),
            ),
            AstNode::Assign("x".to_string(), Box::new(AstNode::from(1_isize))),
            AstNode::Assign("x".to_string(), Box::new(AstNode::from(0_isize))),
        );

        let cost_with_else = estimator.estimate_cost(&if_else_stmt);
        assert!(
            cost_with_else > cost,
            "If-else should have higher cost than simple if"
        );
    }

    #[test]
    fn test_kernel_operations_cost() {
        use crate::ast::helper::*;
        let estimator = OperationCostEstimator;

        // Kernel call should have higher cost than function call
        let func_call = AstNode::CallFunction {
            name: "my_func".to_string(),
            args: vec![AstNode::Var("a".to_string())],
        };

        let kernel_call = call_kernel(
            "my_kernel".to_string(),
            vec![AstNode::Var("a".to_string())],
            [
                Box::new(AstNode::from(100_usize)),
                Box::new(AstNode::from(1_usize)),
                Box::new(AstNode::from(1_usize)),
            ],
            [
                Box::new(AstNode::from(1_usize)),
                Box::new(AstNode::from(1_usize)),
                Box::new(AstNode::from(1_usize)),
            ],
        );

        let func_cost = estimator.estimate_cost(&func_call);
        let kernel_cost = estimator.estimate_cost(&kernel_call);

        assert!(
            kernel_cost > func_cost,
            "Kernel call ({}) should have higher cost than function call ({}) due to launch overhead",
            kernel_cost,
            func_cost
        );
    }

    #[test]
    fn test_bitwise_operations_cost() {
        let estimator = OperationCostEstimator;

        let bitwise_ops = vec![
            AstNode::BitAnd(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            ),
            AstNode::BitOr(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            ),
            AstNode::BitXor(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Var("b".to_string())),
            ),
            AstNode::Shl(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::from(2_usize)),
            ),
            AstNode::Shr(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::from(2_usize)),
            ),
            AstNode::BitNot(Box::new(AstNode::Var("a".to_string()))),
        ];

        for op in bitwise_ops {
            let cost = estimator.estimate_cost(&op);
            assert!(
                cost > 0.0,
                "Bitwise operation {:?} should have non-zero cost",
                op
            );
        }
    }

    #[test]
    fn test_comparison_operations_cost() {
        let estimator = OperationCostEstimator;

        let less_than = AstNode::LessThan(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );

        let eq = AstNode::Eq(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );

        assert!(
            estimator.estimate_cost(&less_than) > 0.0,
            "LessThan should have non-zero cost"
        );
        assert!(
            estimator.estimate_cost(&eq) > 0.0,
            "Eq should have non-zero cost"
        );
    }

    #[test]
    fn test_barrier_cost() {
        let estimator = OperationCostEstimator;
        let barrier = AstNode::Barrier;

        let barrier_cost = estimator.estimate_cost(&barrier);
        let simple_op_cost = estimator.estimate_cost(&AstNode::Add(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        ));

        assert!(
            barrier_cost > simple_op_cost,
            "Barrier ({}) should have higher cost than simple operation ({}) due to synchronization overhead",
            barrier_cost,
            simple_op_cost
        );
    }
}
