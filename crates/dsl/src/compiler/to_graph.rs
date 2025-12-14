//! DSL to Graph compilation

use std::collections::HashMap;

use harp::ast::AstNode;
use harp::ast::helper::wildcard;
use harp::graph::ops::{CumulativeOp, ReduceOp};
use harp::graph::{DType, Graph, GraphNode, GraphOp};

use crate::error::DslError;
use crate::parser::ast::*;

/// Compile a DSL module to Harp Graphs
pub fn compile(module: &DslModule) -> Result<Graph, DslError> {
    // For now, compile the first graph only
    if module.graphs.is_empty() {
        return Err(DslError::CompilationError(
            "No graphs in module".to_string(),
        ));
    }

    compile_graph(&module.graphs[0])
}

/// Compile a single graph definition
fn compile_graph(dsl_graph: &DslGraph) -> Result<Graph, DslError> {
    let mut graph = Graph::new();
    let mut env: HashMap<String, GraphNode> = HashMap::new();

    // Register inputs
    for input in &dsl_graph.inputs {
        let dtype: DType = input.dtype.into();
        let shape: Vec<harp::graph::shape::Expr> =
            input.shape.iter().map(|s| s.to_harp_expr()).collect();
        let node = graph
            .input(&input.name, dtype, shape)
            .with_name(&input.name);
        env.insert(input.name.clone(), node);
    }

    // Compile body statements
    for stmt in &dsl_graph.body {
        compile_statement(stmt, &mut env, &mut graph)?;
    }

    // Register outputs
    for output in &dsl_graph.outputs {
        let node = env.get(&output.name).ok_or_else(|| {
            DslError::UndefinedVariable(format!(
                "Output '{}' not found in environment",
                output.name
            ))
        })?;
        graph.output(&output.name, node.clone());
    }

    Ok(graph)
}

fn compile_statement(
    stmt: &DslStatement,
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
) -> Result<(), DslError> {
    match stmt {
        DslStatement::Let { name, value } | DslStatement::Assign { name, value } => {
            let node = compile_expr(value, env, graph)?.with_name(name);
            env.insert(name.clone(), node);
        }
        DslStatement::Return { .. } => {
            // Return is handled implicitly through outputs
        }
    }
    Ok(())
}

fn compile_expr(
    expr: &DslExpr,
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    match expr {
        DslExpr::Var(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| DslError::UndefinedVariable(format!("Variable '{}' not found", name))),

        DslExpr::IntLit(v) => Ok(GraphNode::from(*v as i32)),

        DslExpr::FloatLit(v) => Ok(GraphNode::from(*v as f32)),

        DslExpr::BinOp { op, lhs, rhs } => {
            let l = compile_expr(lhs, env, graph)?;
            let r = compile_expr(rhs, env, graph)?;
            match op {
                DslBinOp::Add => Ok(&l + &r),
                DslBinOp::Sub => Ok(&l - &r),
                DslBinOp::Mul => Ok(&l * &r),
                DslBinOp::Div => Ok(&l / &r),
                DslBinOp::Rem => Ok(&l % &r),
                _ => Err(DslError::UnsupportedOperation(format!(
                    "Binary op {:?} not yet implemented",
                    op
                ))),
            }
        }

        DslExpr::UnaryOp { op, operand } => {
            let o = compile_expr(operand, env, graph)?;
            match op {
                DslUnaryOp::Neg => Ok(-o),
                DslUnaryOp::Not => Err(DslError::UnsupportedOperation(
                    "Logical not not implemented".to_string(),
                )),
            }
        }

        DslExpr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let recv = compile_expr(receiver, env, graph)?;
            compile_method_call(&recv, method, args, env, graph)
        }

        DslExpr::FunctionCall { name, args } => compile_function_call(name, args, env, graph),

        DslExpr::FusedElementwise { inputs, expr } => {
            compile_fused_elementwise(inputs, expr, env, graph)
        }

        DslExpr::FusedReduce {
            inputs,
            axis,
            op,
            expr,
        } => compile_fused_reduce(inputs, *axis, *op, expr, env, graph),

        DslExpr::FusedCumulative {
            inputs,
            axis,
            op,
            expr,
        } => compile_fused_cumulative(inputs, *axis, *op, expr, env, graph),

        DslExpr::ArrayLit(_) => Err(DslError::UnsupportedOperation(
            "Array literals should be handled in function args".to_string(),
        )),

        DslExpr::Index { .. } => Err(DslError::UnsupportedOperation(
            "Index access not yet implemented".to_string(),
        )),
    }
}

fn compile_method_call(
    receiver: &GraphNode,
    method: &str,
    args: &[DslArg],
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    match method {
        // View operations (use View methods via GraphNode::view)
        "unsqueeze" => {
            let axis = get_axis_arg(args, 0)?;
            let new_view = receiver.view.clone().unsqueeze(axis);
            Ok(receiver.view(new_view))
        }
        "squeeze" => {
            let axis = get_axis_arg(args, 0)?;
            let new_view = receiver.view.clone().squeeze(axis);
            Ok(receiver.view(new_view))
        }
        "permute" => {
            let axes = get_axes_arg(args)?;
            let new_view = receiver.view.clone().permute(axes);
            Ok(receiver.view(new_view))
        }
        "expand" => {
            let shape = get_shape_arg(args, env, graph)?;
            Ok(receiver.expand(shape))
        }
        "reshape" => {
            let shape = get_shape_arg(args, env, graph)?;
            Ok(receiver.reshape(shape))
        }

        // Reduce operations
        "sum" => {
            let axis = get_axis_arg(args, 0)?;
            Ok(receiver.reduce_sum(axis))
        }
        "prod" => {
            let axis = get_axis_arg(args, 0)?;
            Ok(receiver.reduce_mul(axis))
        }
        "max" => {
            let axis = get_axis_arg(args, 0)?;
            Ok(receiver.reduce_max(axis))
        }
        "mean" => {
            let axis = get_axis_arg(args, 0)?;
            Ok(receiver.clone().mean(axis))
        }

        // Cumulative operations
        "cumsum" => {
            let axis = get_axis_arg(args, 0)?;
            Ok(receiver.cumsum(axis))
        }
        "cumprod" => {
            let axis = get_axis_arg(args, 0)?;
            Ok(receiver.cumprod(axis))
        }

        // Math operations
        "log2" => Ok(receiver.clone().log2()),
        "exp2" => Ok(receiver.clone().exp2()),
        "sqrt" => Ok(receiver.clone().sqrt()),
        "rsqrt" => Ok(receiver.clone().rsqrt()),
        "sin" => Ok(receiver.clone().sin()),
        "cos" => Ok(receiver.clone().cos()),
        "recip" => Ok(receiver.clone().recip()),
        "square" => Ok(receiver.clone().square()),
        "abs_square" => Ok(receiver.clone().abs_square()),

        _ => Err(DslError::UnsupportedOperation(format!(
            "Method '{}' not implemented",
            method
        ))),
    }
}

fn compile_function_call(
    name: &str,
    args: &[DslArg],
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    match name {
        "matmul" => {
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "matmul requires 2 arguments".to_string(),
                ));
            }
            let a = get_positional_expr_arg(&args[0], env, graph)?;
            let b = get_positional_expr_arg(&args[1], env, graph)?;
            Ok(a.matmul(b))
        }
        "concat" => {
            let (tensors, axis) = get_concat_args(args, env, graph)?;
            Ok(harp::graph::ops::concat(tensors, axis))
        }
        "max" => {
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "max requires 2 arguments".to_string(),
                ));
            }
            let a = get_positional_expr_arg(&args[0], env, graph)?;
            let b = get_positional_expr_arg(&args[1], env, graph)?;
            Ok(a.max(b))
        }
        "min" => {
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "min requires 2 arguments".to_string(),
                ));
            }
            let a = get_positional_expr_arg(&args[0], env, graph)?;
            let b = get_positional_expr_arg(&args[1], env, graph)?;
            Ok(a.min(b))
        }
        _ => Err(DslError::UnsupportedOperation(format!(
            "Function '{}' not implemented",
            name
        ))),
    }
}

fn compile_fused_elementwise(
    inputs: &[String],
    expr: &DslExpr,
    env: &HashMap<String, GraphNode>,
    _graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    // Collect input nodes
    let input_nodes: Vec<GraphNode> = inputs
        .iter()
        .map(|name| {
            env.get(name)
                .cloned()
                .ok_or_else(|| DslError::UndefinedVariable(name.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Build AST expression using wildcards
    let ast_expr = build_fused_ast_expr(expr, inputs)?;

    // Create FusedElementwise node
    let dtype = input_nodes[0].dtype.clone();
    let view = input_nodes[0].view.clone();

    Ok(GraphNode::new(
        dtype,
        GraphOp::FusedElementwise { expr: ast_expr },
        input_nodes,
        view,
    ))
}

fn compile_fused_reduce(
    inputs: &[String],
    axis: usize,
    op: ReduceOpKind,
    expr: &DslExpr,
    env: &HashMap<String, GraphNode>,
    _graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    let input_nodes: Vec<GraphNode> = inputs
        .iter()
        .map(|name| {
            env.get(name)
                .cloned()
                .ok_or_else(|| DslError::UndefinedVariable(name.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let ast_expr = build_fused_ast_expr(expr, inputs)?;

    let reduce_op = match op {
        ReduceOpKind::Sum => ReduceOp::Sum,
        ReduceOpKind::Prod => ReduceOp::Prod,
        ReduceOpKind::Max => ReduceOp::Max,
    };

    let dtype = input_nodes[0].dtype.clone();
    let view = input_nodes[0].view.clone();
    let mut new_shape = view.shape().to_vec();
    new_shape.remove(axis);
    let reduced_view = harp::graph::shape::View::contiguous(new_shape);

    Ok(GraphNode::new(
        dtype,
        GraphOp::FusedElementwiseReduce {
            expr: ast_expr,
            reduce_op,
            axes: vec![axis],
            reduce_strategy: None,
        },
        input_nodes,
        reduced_view,
    ))
}

fn compile_fused_cumulative(
    inputs: &[String],
    axis: usize,
    op: CumulativeOpKind,
    expr: &DslExpr,
    env: &HashMap<String, GraphNode>,
    _graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    let input_nodes: Vec<GraphNode> = inputs
        .iter()
        .map(|name| {
            env.get(name)
                .cloned()
                .ok_or_else(|| DslError::UndefinedVariable(name.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let ast_expr = build_fused_ast_expr(expr, inputs)?;

    let cumulative_op = match op {
        CumulativeOpKind::Sum => CumulativeOp::Sum,
        CumulativeOpKind::Prod => CumulativeOp::Prod,
    };

    let dtype = input_nodes[0].dtype.clone();
    let view = input_nodes[0].view.clone();

    Ok(GraphNode::new(
        dtype,
        GraphOp::FusedElementwiseCumulative {
            expr: ast_expr,
            cumulative_op,
            axis,
            cumulative_strategy: None,
        },
        input_nodes,
        view,
    ))
}

/// Build AST expression for fused operations, replacing variable references with wildcards
fn build_fused_ast_expr(expr: &DslExpr, inputs: &[String]) -> Result<AstNode, DslError> {
    match expr {
        DslExpr::Var(name) => {
            // Find index in inputs
            if let Some(idx) = inputs.iter().position(|n| n == name) {
                Ok(wildcard(idx.to_string()))
            } else {
                Err(DslError::UndefinedVariable(format!(
                    "Variable '{}' not in fused inputs",
                    name
                )))
            }
        }
        DslExpr::IntLit(v) => Ok(AstNode::Const(harp::ast::Literal::Int(*v as isize))),
        DslExpr::FloatLit(v) => Ok(AstNode::Const(harp::ast::Literal::F32(*v as f32))),
        DslExpr::BinOp { op, lhs, rhs } => {
            let l = build_fused_ast_expr(lhs, inputs)?;
            let r = build_fused_ast_expr(rhs, inputs)?;
            match op {
                DslBinOp::Add => Ok(l + r),
                DslBinOp::Sub => Ok(l - r),
                DslBinOp::Mul => Ok(l * r),
                DslBinOp::Div => Ok(l / r),
                DslBinOp::Rem => Ok(l % r),
                _ => Err(DslError::UnsupportedOperation(format!(
                    "Binary op {:?} not supported in fused expr",
                    op
                ))),
            }
        }
        DslExpr::UnaryOp { op, operand } => {
            let o = build_fused_ast_expr(operand, inputs)?;
            match op {
                DslUnaryOp::Neg => Ok(-o),
                DslUnaryOp::Not => Err(DslError::UnsupportedOperation(
                    "Not not supported in fused expr".to_string(),
                )),
            }
        }
        _ => Err(DslError::UnsupportedOperation(format!(
            "Expression type not supported in fused expr: {:?}",
            expr
        ))),
    }
}

// Helper functions for argument extraction

fn get_axis_arg(args: &[DslArg], default: usize) -> Result<usize, DslError> {
    if args.is_empty() {
        return Ok(default);
    }
    match &args[0] {
        DslArg::Positional(DslExpr::IntLit(v)) => Ok(*v as usize),
        DslArg::Named {
            name,
            value: DslArgValue::Expr(DslExpr::IntLit(v)),
        } if name == "axis" => Ok(*v as usize),
        _ => Err(DslError::InvalidArgument(
            "Expected axis as integer".to_string(),
        )),
    }
}

fn get_axes_arg(args: &[DslArg]) -> Result<Vec<usize>, DslError> {
    if args.is_empty() {
        return Err(DslError::InvalidArgument("Expected axes".to_string()));
    }
    match &args[0] {
        DslArg::Array(exprs) => exprs
            .iter()
            .map(|e| match e {
                DslExpr::IntLit(v) => Ok(*v as usize),
                _ => Err(DslError::InvalidArgument(
                    "Expected integer in axes".to_string(),
                )),
            })
            .collect(),
        _ => Err(DslError::InvalidArgument(
            "Expected array of axes".to_string(),
        )),
    }
}

fn get_shape_arg(
    args: &[DslArg],
    _env: &HashMap<String, GraphNode>,
    _graph: &mut Graph,
) -> Result<Vec<harp::graph::shape::Expr>, DslError> {
    if args.is_empty() {
        return Err(DslError::InvalidArgument("Expected shape".to_string()));
    }
    match &args[0] {
        DslArg::Array(exprs) => exprs
            .iter()
            .map(|e| match e {
                DslExpr::IntLit(v) => Ok(harp::graph::shape::Expr::Const(*v as isize)),
                DslExpr::Var(name) => Ok(harp::graph::shape::Expr::Var(name.clone())),
                _ => Err(DslError::InvalidArgument(
                    "Expected integer or var in shape".to_string(),
                )),
            })
            .collect(),
        _ => Err(DslError::InvalidArgument(
            "Expected array shape".to_string(),
        )),
    }
}

fn get_positional_expr_arg(
    arg: &DslArg,
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    match arg {
        DslArg::Positional(expr) => compile_expr(expr, env, graph),
        _ => Err(DslError::InvalidArgument(
            "Expected positional expression argument".to_string(),
        )),
    }
}

fn get_concat_args(
    args: &[DslArg],
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
) -> Result<(Vec<GraphNode>, usize), DslError> {
    if args.len() != 2 {
        return Err(DslError::InvalidArgument(
            "concat requires array and axis".to_string(),
        ));
    }

    let tensors = match &args[0] {
        DslArg::Array(exprs) => exprs
            .iter()
            .map(|e| compile_expr(e, env, graph))
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(DslError::InvalidArgument(
                "First argument to concat must be array".to_string(),
            ));
        }
    };

    let axis = match &args[1] {
        DslArg::Positional(DslExpr::IntLit(v)) => *v as usize,
        DslArg::Named {
            name,
            value: DslArgValue::Expr(DslExpr::IntLit(v)),
        } if name == "axis" => *v as usize,
        _ => {
            return Err(DslError::InvalidArgument(
                "Second argument to concat must be axis".to_string(),
            ));
        }
    };

    Ok((tensors, axis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn test_compile_simple_add() {
        let source = r#"
            graph add(a: f32[N, M], b: f32[N, M]) -> (c: f32[N, M]) {
                c = a + b
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.input_metas().len(), 2);
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_compile_method_chain() {
        let source = r#"
            graph test(x: f32[N, M]) -> (y: f32[1, N, M]) {
                let a = x.unsqueeze(0)
                y = a
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.input_metas().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
    }
}
