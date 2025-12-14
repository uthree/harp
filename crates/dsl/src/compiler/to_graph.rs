//! DSL to Graph compilation

use std::collections::HashMap;

use harp::ast::AstNode;
use harp::ast::helper::wildcard;
use harp::graph::ops::{CumulativeOp, ReduceOp};
use harp::graph::{DType, Graph, GraphNode, GraphOp, View};

use crate::error::DslError;
use crate::parser::ast::*;

/// Default maximum recursion depth for subgraph calls
pub const DEFAULT_MAX_RECURSION_DEPTH: usize = 10;

/// Compilation context for tracking subgraphs and recursion
pub struct CompileContext<'a> {
    /// The module being compiled
    module: &'a DslModule,
    /// Call stack for recursion detection (graph names)
    call_stack: Vec<String>,
    /// Maximum recursion depth
    max_recursion_depth: usize,
    /// Cache of compiled subgraphs
    compiled_subgraphs: HashMap<String, Graph>,
}

impl<'a> CompileContext<'a> {
    /// Create a new compilation context
    pub fn new(module: &'a DslModule) -> Self {
        Self {
            module,
            call_stack: Vec::new(),
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            compiled_subgraphs: HashMap::new(),
        }
    }

    /// Check if calling a graph would exceed recursion depth
    fn check_recursion(&self, graph_name: &str) -> Result<(), DslError> {
        let depth = self
            .call_stack
            .iter()
            .filter(|&name| name == graph_name)
            .count();
        if depth >= self.max_recursion_depth {
            return Err(DslError::RecursionDepthExceeded {
                graph_name: graph_name.to_string(),
                depth,
                max_depth: self.max_recursion_depth,
            });
        }
        Ok(())
    }

    /// Push a graph name onto the call stack
    fn push_call(&mut self, graph_name: &str) {
        self.call_stack.push(graph_name.to_string());
    }

    /// Pop a graph name from the call stack
    fn pop_call(&mut self) {
        self.call_stack.pop();
    }

    /// Find a graph definition by name
    fn find_graph(&self, name: &str) -> Option<&'a DslGraph> {
        self.module.graphs.iter().find(|g| g.name == name)
    }

    /// Check if a name is a subgraph (exists in module and is not a builtin)
    fn is_subgraph(&self, name: &str) -> bool {
        // Check if it's a builtin function first
        let builtins = ["matmul", "concat", "max", "min"];
        if builtins.contains(&name) {
            return false;
        }
        // Check if it exists in the module
        self.find_graph(name).is_some()
    }
}

/// Compile a DSL module to a Harp Graph
///
/// The module must contain a graph named "main" which serves as the entry point.
/// All other graphs are treated as subgraphs and can be called from main.
pub fn compile(module: &DslModule) -> Result<Graph, DslError> {
    if module.graphs.is_empty() {
        return Err(DslError::CompilationError(
            "No graphs in module".to_string(),
        ));
    }

    // Find the main graph
    let main_graph = module
        .graphs
        .iter()
        .find(|g| g.name == "main")
        .ok_or(DslError::NoMainGraph)?;

    // Create compilation context
    let mut ctx = CompileContext::new(module);

    // Compile the main graph
    compile_graph_with_context(main_graph, &mut ctx)
}

/// Compile a single graph definition with context
fn compile_graph_with_context(
    dsl_graph: &DslGraph,
    ctx: &mut CompileContext,
) -> Result<Graph, DslError> {
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
        compile_statement_with_context(stmt, &mut env, &mut graph, ctx)?;
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

    // Copy compiled subgraphs to the main graph
    for (name, subgraph) in ctx.compiled_subgraphs.iter() {
        graph.add_subgraph(name.clone(), subgraph.clone());
    }

    Ok(graph)
}

/// Compile a single graph definition (for subgraphs, without adding subgraphs to result)
fn compile_subgraph(dsl_graph: &DslGraph, ctx: &mut CompileContext) -> Result<Graph, DslError> {
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
        compile_statement_with_context(stmt, &mut env, &mut graph, ctx)?;
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

fn compile_statement_with_context(
    stmt: &DslStatement,
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
    ctx: &mut CompileContext,
) -> Result<(), DslError> {
    match stmt {
        DslStatement::Let { name, value } | DslStatement::Assign { name, value } => {
            let node = compile_expr_with_context(value, env, graph, ctx)?.with_name(name);
            env.insert(name.clone(), node);
        }
        DslStatement::TupleLet { names, value } => {
            // Tuple unpacking: let (a, b, c) = subgraph_call(...)
            compile_tuple_let(names, value, env, graph, ctx)?;
        }
        DslStatement::Return { .. } => {
            // Return is handled implicitly through outputs
        }
    }
    Ok(())
}

/// Compile a tuple let statement: let (a, b, c) = subgraph_call(...)
fn compile_tuple_let(
    names: &[String],
    value: &DslExpr,
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
    ctx: &mut CompileContext,
) -> Result<(), DslError> {
    // The value must be a function call to a subgraph with multiple outputs
    match value {
        DslExpr::FunctionCall { name, args } => {
            // Check if it's a subgraph
            if !ctx.is_subgraph(name) {
                return Err(DslError::CompilationError(format!(
                    "Tuple unpacking is only supported for subgraph calls, '{}' is not a subgraph",
                    name
                )));
            }

            // Get the subgraph definition to check output count
            let subgraph_def = ctx
                .find_graph(name)
                .ok_or_else(|| DslError::UndefinedFunction(name.to_string()))?
                .clone();

            // Check that output count matches tuple size
            if subgraph_def.outputs.len() != names.len() {
                return Err(DslError::CompilationError(format!(
                    "Subgraph '{}' has {} outputs, but tuple has {} elements",
                    name,
                    subgraph_def.outputs.len(),
                    names.len()
                )));
            }

            // Check recursion depth
            ctx.check_recursion(name)?;

            // Check argument count
            if args.len() != subgraph_def.inputs.len() {
                return Err(DslError::SubGraphArgumentMismatch {
                    graph_name: name.to_string(),
                    expected: subgraph_def.inputs.len(),
                    got: args.len(),
                });
            }

            // Compile arguments
            let mut input_nodes = Vec::new();
            for arg in args.iter() {
                let node = get_positional_expr_arg_with_context(arg, env, graph, ctx)?;
                input_nodes.push(node);
            }

            // Compile the subgraph if not already cached
            if !ctx.compiled_subgraphs.contains_key(name) {
                ctx.push_call(name);
                let compiled = compile_subgraph(&subgraph_def, ctx)?;
                ctx.pop_call();
                ctx.compiled_subgraphs.insert(name.to_string(), compiled);
            }

            // Create the SubGraphCall node (using the first output's type for the call itself)
            let first_output = &subgraph_def.outputs[0];
            let first_dtype: DType = first_output.dtype.into();
            let first_shape: Vec<harp::graph::shape::Expr> = first_output
                .shape
                .iter()
                .map(|s| s.to_harp_expr())
                .collect();
            let first_view = View::contiguous(first_shape);

            let call_node = GraphNode::subgraph_call(name, input_nodes, first_dtype, first_view);

            // Create SubGraphOutput nodes for each output and bind them to names
            for (i, (var_name, output)) in names.iter().zip(subgraph_def.outputs.iter()).enumerate()
            {
                let output_dtype: DType = output.dtype.into();
                let output_shape: Vec<harp::graph::shape::Expr> =
                    output.shape.iter().map(|s| s.to_harp_expr()).collect();
                let output_view = View::contiguous(output_shape);

                let output_node = GraphNode::subgraph_output(
                    call_node.clone(),
                    i,
                    &output.name,
                    output_dtype,
                    output_view,
                )
                .with_name(var_name);

                env.insert(var_name.clone(), output_node);
            }

            Ok(())
        }
        _ => Err(DslError::CompilationError(
            "Tuple unpacking is only supported for function calls (subgraph calls)".to_string(),
        )),
    }
}

fn compile_expr_with_context(
    expr: &DslExpr,
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
    ctx: &mut CompileContext,
) -> Result<GraphNode, DslError> {
    match expr {
        DslExpr::Var(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| DslError::UndefinedVariable(format!("Variable '{}' not found", name))),

        DslExpr::IntLit(v) => Ok(GraphNode::from(*v as i32)),

        DslExpr::FloatLit(v) => Ok(GraphNode::from(*v as f32)),

        DslExpr::BinOp { op, lhs, rhs } => {
            let l = compile_expr_with_context(lhs, env, graph, ctx)?;
            let r = compile_expr_with_context(rhs, env, graph, ctx)?;
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
            let o = compile_expr_with_context(operand, env, graph, ctx)?;
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
            let recv = compile_expr_with_context(receiver, env, graph, ctx)?;
            compile_method_call(&recv, method, args, env, graph, ctx)
        }

        DslExpr::FunctionCall { name, args } => {
            compile_function_call_with_context(name, args, env, graph, ctx)
        }

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
    _ctx: &mut CompileContext,
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

fn compile_function_call_with_context(
    name: &str,
    args: &[DslArg],
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
    ctx: &mut CompileContext,
) -> Result<GraphNode, DslError> {
    // Check if it's a builtin function
    match name {
        "matmul" => {
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "matmul requires 2 arguments".to_string(),
                ));
            }
            let a = get_positional_expr_arg_with_context(&args[0], env, graph, ctx)?;
            let b = get_positional_expr_arg_with_context(&args[1], env, graph, ctx)?;
            return Ok(a.matmul(b));
        }
        "concat" => {
            let (tensors, axis) = get_concat_args_with_context(args, env, graph, ctx)?;
            return Ok(harp::graph::ops::concat(tensors, axis));
        }
        "max" => {
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "max requires 2 arguments".to_string(),
                ));
            }
            let a = get_positional_expr_arg_with_context(&args[0], env, graph, ctx)?;
            let b = get_positional_expr_arg_with_context(&args[1], env, graph, ctx)?;
            return Ok(a.max(b));
        }
        "min" => {
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "min requires 2 arguments".to_string(),
                ));
            }
            let a = get_positional_expr_arg_with_context(&args[0], env, graph, ctx)?;
            let b = get_positional_expr_arg_with_context(&args[1], env, graph, ctx)?;
            return Ok(a.min(b));
        }
        _ => {}
    }

    // Check if it's a subgraph
    if ctx.is_subgraph(name) {
        return compile_subgraph_call(name, args, env, graph, ctx);
    }

    // Unknown function
    Err(DslError::UndefinedFunction(name.to_string()))
}

/// Compile a subgraph call
fn compile_subgraph_call(
    name: &str,
    args: &[DslArg],
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
    ctx: &mut CompileContext,
) -> Result<GraphNode, DslError> {
    // Check recursion depth
    ctx.check_recursion(name)?;

    // Find the subgraph definition
    let subgraph_def = ctx
        .find_graph(name)
        .ok_or_else(|| DslError::UndefinedFunction(name.to_string()))?
        .clone(); // Clone to avoid borrow issues

    // Check argument count
    if args.len() != subgraph_def.inputs.len() {
        return Err(DslError::SubGraphArgumentMismatch {
            graph_name: name.to_string(),
            expected: subgraph_def.inputs.len(),
            got: args.len(),
        });
    }

    // Compile arguments (inputs to the subgraph)
    let mut input_nodes = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        let node = get_positional_expr_arg_with_context(arg, env, graph, ctx)?;
        // TODO: Add type checking here
        let _expected_param = &subgraph_def.inputs[i];
        input_nodes.push(node);
    }

    // Compile the subgraph if not already cached
    if !ctx.compiled_subgraphs.contains_key(name) {
        ctx.push_call(name);
        let compiled = compile_subgraph(&subgraph_def, ctx)?;
        ctx.pop_call();
        ctx.compiled_subgraphs.insert(name.to_string(), compiled);
    }

    // Get output information from the subgraph definition
    let output_count = subgraph_def.outputs.len();

    if output_count == 1 {
        // Single output: return SubGraphCall directly
        let output = &subgraph_def.outputs[0];
        let output_dtype: DType = output.dtype.into();
        let output_shape: Vec<harp::graph::shape::Expr> =
            output.shape.iter().map(|s| s.to_harp_expr()).collect();
        let output_view = View::contiguous(output_shape);

        Ok(GraphNode::subgraph_call(
            name,
            input_nodes,
            output_dtype,
            output_view,
        ))
    } else {
        // Multiple outputs: This case requires tuple unpacking syntax
        // For now, return the first output as the default
        // TODO: Implement proper tuple unpacking support
        let output = &subgraph_def.outputs[0];
        let output_dtype: DType = output.dtype.into();
        let output_shape: Vec<harp::graph::shape::Expr> =
            output.shape.iter().map(|s| s.to_harp_expr()).collect();
        let output_view = View::contiguous(output_shape);

        let call_node =
            GraphNode::subgraph_call(name, input_nodes, output_dtype.clone(), output_view.clone());

        // Return SubGraphOutput for the first output
        Ok(GraphNode::subgraph_output(
            call_node,
            0,
            &output.name,
            output_dtype,
            output_view,
        ))
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

/// Get axis argument from args
/// Supports: positional (0), keyword (axis=0, dim=0)
/// Returns None if no args provided (meaning "all axes")
fn get_optional_axis_arg(args: &[DslArg]) -> Result<Option<usize>, DslError> {
    if args.is_empty() {
        return Ok(None);
    }
    match &args[0] {
        DslArg::Positional(DslExpr::IntLit(v)) => Ok(Some(*v as usize)),
        DslArg::Named {
            name,
            value: DslArgValue::Expr(DslExpr::IntLit(v)),
        } if name == "axis" || name == "dim" => Ok(Some(*v as usize)),
        _ => Err(DslError::InvalidArgument(
            "Expected axis/dim as integer".to_string(),
        )),
    }
}

fn get_axis_arg(args: &[DslArg], default: usize) -> Result<usize, DslError> {
    get_optional_axis_arg(args).map(|opt| opt.unwrap_or(default))
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

fn get_positional_expr_arg_with_context(
    arg: &DslArg,
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
    ctx: &mut CompileContext,
) -> Result<GraphNode, DslError> {
    match arg {
        DslArg::Positional(expr) => compile_expr_with_context(expr, env, graph, ctx),
        _ => Err(DslError::InvalidArgument(
            "Expected positional expression argument".to_string(),
        )),
    }
}

fn get_concat_args_with_context(
    args: &[DslArg],
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
    ctx: &mut CompileContext,
) -> Result<(Vec<GraphNode>, usize), DslError> {
    if args.len() != 2 {
        return Err(DslError::InvalidArgument(
            "concat requires array and axis".to_string(),
        ));
    }

    let tensors = match &args[0] {
        DslArg::Array(exprs) => exprs
            .iter()
            .map(|e| compile_expr_with_context(e, env, graph, ctx))
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

// ============================================================================
// Legacy API (for backward compatibility)
// ============================================================================
//
// These functions are kept for backward compatibility but are not actively used
// in the crate. They compile single graphs without subgraph support.

/// Compile a DSL module to Harp Graphs (legacy API)
///
/// This function maintains backward compatibility by compiling the first graph
/// if no "main" graph exists.
#[allow(dead_code)]
pub fn compile_legacy(module: &DslModule) -> Result<Graph, DslError> {
    if module.graphs.is_empty() {
        return Err(DslError::CompilationError(
            "No graphs in module".to_string(),
        ));
    }

    // Try to find main first, fall back to first graph
    if module.graphs.iter().any(|g| g.name == "main") {
        compile(module)
    } else {
        compile_graph(&module.graphs[0])
    }
}

/// Compile a single graph definition (legacy, without subgraph support)
#[allow(dead_code)]
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

#[allow(dead_code)]
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
        DslStatement::TupleLet { .. } => {
            // Tuple unpacking is not supported in legacy API (no subgraph support)
            return Err(DslError::UnsupportedOperation(
                "Tuple unpacking requires subgraph support, use compile() instead of compile_legacy()".to_string(),
            ));
        }
        DslStatement::Return { .. } => {
            // Return is handled implicitly through outputs
        }
    }
    Ok(())
}

#[allow(dead_code)]
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
            compile_method_call_legacy(&recv, method, args, env, graph)
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

#[allow(dead_code)]
fn compile_method_call_legacy(
    receiver: &GraphNode,
    method: &str,
    args: &[DslArg],
    env: &mut HashMap<String, GraphNode>,
    graph: &mut Graph,
) -> Result<GraphNode, DslError> {
    match method {
        // View operations
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
            graph main(a: f32[N, M], b: f32[N, M]) -> (c: f32[N, M]) {
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
            graph main(x: f32[N, M]) -> (y: f32[1, N, M]) {
                let a = x.unsqueeze(0)
                y = a
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.input_metas().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_compile_subgraph_call() {
        let source = r#"
            graph relu(x: f32[N, M]) -> (y: f32[N, M]) {
                let zero = 0.0
                y = max(x, zero)
            }

            graph main(input: f32[B, 256]) -> (output: f32[B, 256]) {
                output = relu(input)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.input_metas().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
        // Check that subgraph is registered
        assert!(graph.subgraph("relu").is_some());
    }

    #[test]
    fn test_compile_nested_subgraph() {
        let source = r#"
            graph square(x: f32[N]) -> (y: f32[N]) {
                y = x * x
            }

            graph quad(x: f32[N]) -> (y: f32[N]) {
                let sq = square(x)
                y = square(sq)
            }

            graph main(input: f32[10]) -> (output: f32[10]) {
                output = quad(input)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.input_metas().len(), 1);
        assert!(graph.subgraph("square").is_some());
        assert!(graph.subgraph("quad").is_some());
    }

    #[test]
    fn test_no_main_graph_error() {
        let source = r#"
            graph add(a: f32[N], b: f32[N]) -> (c: f32[N]) {
                c = a + b
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(matches!(result, Err(DslError::NoMainGraph)));
    }

    #[test]
    fn test_undefined_subgraph_error() {
        let source = r#"
            graph main(input: f32[N]) -> (output: f32[N]) {
                output = unknown_func(input)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(matches!(result, Err(DslError::UndefinedFunction(_))));
    }

    #[test]
    fn test_subgraph_argument_mismatch() {
        let source = r#"
            graph add(a: f32[N], b: f32[N]) -> (c: f32[N]) {
                c = a + b
            }

            graph main(input: f32[10]) -> (output: f32[10]) {
                output = add(input)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(matches!(
            result,
            Err(DslError::SubGraphArgumentMismatch { .. })
        ));
    }

    #[test]
    fn test_compile_keyword_arg_dim() {
        let source = r#"
            graph main(x: f32[N, M]) -> (y: f32[N]) {
                y = x.sum(dim=1)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_compile_keyword_arg_axis() {
        let source = r#"
            graph main(x: f32[N, M]) -> (y: f32[N]) {
                y = x.sum(axis=1)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_compile_positional_arg() {
        let source = r#"
            graph main(x: f32[N, M]) -> (y: f32[N]) {
                y = x.sum(1)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_tuple_unpacking_multi_output() {
        let source = r#"
            graph split_add_mul(x: f32[N]) -> (added: f32[N], multiplied: f32[N]) {
                let one = 1.0
                let two = 2.0
                added = x + one
                multiplied = x * two
            }

            graph main(input: f32[10]) -> (out1: f32[10], out2: f32[10]) {
                let (a, b) = split_add_mul(input)
                out1 = a
                out2 = b
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.input_metas().len(), 1);
        assert_eq!(graph.outputs().len(), 2);
        assert!(graph.subgraph("split_add_mul").is_some());
    }

    #[test]
    fn test_tuple_unpacking_mismatch_error() {
        let source = r#"
            graph two_outputs(x: f32[N]) -> (a: f32[N], b: f32[N]) {
                a = x
                b = x
            }

            graph main(input: f32[10]) -> (out: f32[10]) {
                let (x, y, z) = two_outputs(input)
                out = x
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("2 outputs") && err_str.contains("3 elements"));
    }
}
