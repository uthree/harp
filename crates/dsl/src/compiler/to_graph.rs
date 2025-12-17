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

    // Register shape variable defaults
    for (var_name, default_value) in &dsl_graph.shape_vars {
        graph.set_shape_var_default(var_name.clone(), *default_value);
    }

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

    // Find and validate return statement
    let return_names = find_return_statement(&dsl_graph.body, &dsl_graph.name)?;

    // Validate return count matches output count
    if return_names.len() != dsl_graph.outputs.len() {
        return Err(DslError::CompilationError(format!(
            "Graph '{}' has {} outputs but return statement has {} values",
            dsl_graph.name,
            dsl_graph.outputs.len(),
            return_names.len()
        )));
    }

    // Compile body statements (excluding return, which is handled separately)
    for stmt in &dsl_graph.body {
        compile_statement_with_context(stmt, &mut env, &mut graph, ctx)?;
    }

    // Register outputs using return statement names paired with output parameter names
    for (return_name, output) in return_names.iter().zip(dsl_graph.outputs.iter()) {
        let node = env.get(return_name).ok_or_else(|| {
            DslError::UndefinedVariable(format!(
                "Return value '{}' not found in graph '{}'",
                return_name, dsl_graph.name
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

/// Find and validate return statement in graph body
fn find_return_statement(body: &[DslStatement], graph_name: &str) -> Result<Vec<String>, DslError> {
    // Find return statement (should be at the end, but we search all)
    let return_stmt = body.iter().find_map(|stmt| {
        if let DslStatement::Return { names } = stmt {
            Some(names.clone())
        } else {
            None
        }
    });

    return_stmt.ok_or_else(|| {
        DslError::CompilationError(format!(
            "Graph '{}' is missing a return statement",
            graph_name
        ))
    })
}

/// Compile a single graph definition (for subgraphs, without adding subgraphs to result)
fn compile_subgraph(dsl_graph: &DslGraph, ctx: &mut CompileContext) -> Result<Graph, DslError> {
    let mut graph = Graph::new();
    let mut env: HashMap<String, GraphNode> = HashMap::new();

    // Register shape variable defaults
    for (var_name, default_value) in &dsl_graph.shape_vars {
        graph.set_shape_var_default(var_name.clone(), *default_value);
    }

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

    // Find and validate return statement
    let return_names = find_return_statement(&dsl_graph.body, &dsl_graph.name)?;

    // Validate return count matches output count
    if return_names.len() != dsl_graph.outputs.len() {
        return Err(DslError::CompilationError(format!(
            "Graph '{}' has {} outputs but return statement has {} values",
            dsl_graph.name,
            dsl_graph.outputs.len(),
            return_names.len()
        )));
    }

    // Compile body statements
    for stmt in &dsl_graph.body {
        compile_statement_with_context(stmt, &mut env, &mut graph, ctx)?;
    }

    // Register outputs using return statement names paired with output parameter names
    for (return_name, output) in return_names.iter().zip(dsl_graph.outputs.iter()) {
        let node = env.get(return_name).ok_or_else(|| {
            DslError::UndefinedVariable(format!(
                "Return value '{}' not found in graph '{}'",
                return_name, dsl_graph.name
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
        DslStatement::Assign { name, value } => {
            let node = compile_expr_with_context(value, env, graph, ctx)?.with_name(name);
            env.insert(name.clone(), node);
        }
        DslStatement::TupleAssign { names, value } => {
            // Tuple unpacking: (a, b, c) = subgraph_call(...)
            compile_tuple_assign(names, value, env, graph, ctx)?;
        }
        DslStatement::Return { .. } => {
            // Return is handled implicitly through outputs
        }
    }
    Ok(())
}

/// Compile a tuple assignment: (a, b, c) = subgraph_call(...)
fn compile_tuple_assign(
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
                return Err(DslError::SubgraphArgumentMismatch {
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

            // Create the SubgraphCall node (using the first output's type for the call itself)
            let first_output = &subgraph_def.outputs[0];
            let first_dtype: DType = first_output.dtype.into();
            let first_shape: Vec<harp::graph::shape::Expr> = first_output
                .shape
                .iter()
                .map(|s| s.to_harp_expr())
                .collect();
            let first_view = View::contiguous(first_shape);

            let call_node = GraphNode::subgraph_call(name, input_nodes, first_dtype, first_view);

            // Create SubgraphOutput nodes for each output and bind them to names
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
        "repeat" => {
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "repeat requires 2 arguments: axis and times".to_string(),
                ));
            }
            let axis = get_axis_arg(args, 0)?;
            let times = get_expr_arg(&args[1])?;
            Ok(receiver.repeat(axis, times))
        }
        "reshape" => {
            let shape = get_shape_arg(args, env, graph)?;
            Ok(receiver.reshape(shape))
        }
        "view_expr" => {
            // view_expr([shape], index_expr)
            // index_expr can contain idx0, idx1, etc. which are converted to Expr::Idx
            if args.len() != 2 {
                return Err(DslError::InvalidArgument(
                    "view_expr requires 2 arguments: shape array and index expression".to_string(),
                ));
            }
            let shape = get_shape_arg(args, env, graph)?;
            let index_expr = get_index_expr_arg(&args[1])?;
            let new_view = View::from_index_expr(shape, index_expr);
            Ok(receiver.view(new_view))
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
        return Err(DslError::SubgraphArgumentMismatch {
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
        // Single output: return SubgraphCall directly
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

        // Return SubgraphOutput for the first output
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

/// Get a single Expr argument (for times parameter in repeat)
fn get_expr_arg(arg: &DslArg) -> Result<harp::graph::shape::Expr, DslError> {
    match arg {
        DslArg::Positional(expr) => match expr {
            DslExpr::IntLit(v) => Ok(harp::graph::shape::Expr::Const(*v as isize)),
            DslExpr::Var(name) => Ok(harp::graph::shape::Expr::Var(name.clone())),
            _ => Err(DslError::InvalidArgument(
                "Expected integer or var for times argument".to_string(),
            )),
        },
        _ => Err(DslError::InvalidArgument(
            "Expected positional argument for times".to_string(),
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

/// Get an index expression argument for view_expr
/// Converts idx0, idx1, etc. to Expr::Idx(i)
fn get_index_expr_arg(arg: &DslArg) -> Result<harp::graph::shape::Expr, DslError> {
    match arg {
        DslArg::Positional(expr) => dsl_expr_to_index_expr(expr),
        _ => Err(DslError::InvalidArgument(
            "Expected positional expression for index_expr".to_string(),
        )),
    }
}

/// Convert DSL expression to Harp shape Expr, converting idx0, idx1, etc. to Expr::Idx
fn dsl_expr_to_index_expr(expr: &DslExpr) -> Result<harp::graph::shape::Expr, DslError> {
    use harp::graph::shape::Expr;

    match expr {
        DslExpr::IntLit(v) => Ok(Expr::Const(*v as isize)),
        DslExpr::Var(name) => {
            // Check if the name matches idx0, idx1, idx2, etc.
            if name.starts_with("idx") {
                if let Ok(i) = name[3..].parse::<usize>() {
                    return Ok(Expr::Idx(i));
                }
            }
            // Otherwise treat as a shape variable
            Ok(Expr::Var(name.clone()))
        }
        DslExpr::BinOp { op, lhs, rhs } => {
            let l = dsl_expr_to_index_expr(lhs)?;
            let r = dsl_expr_to_index_expr(rhs)?;
            match op {
                DslBinOp::Add => Ok(l + r),
                DslBinOp::Sub => Ok(l - r),
                DslBinOp::Mul => Ok(l * r),
                DslBinOp::Div => Ok(l / r),
                DslBinOp::Rem => Ok(l % r),
                _ => Err(DslError::UnsupportedOperation(format!(
                    "Binary op {:?} not supported in index expression",
                    op
                ))),
            }
        }
        DslExpr::UnaryOp { op, operand } => {
            let o = dsl_expr_to_index_expr(operand)?;
            match op {
                DslUnaryOp::Neg => Ok(Expr::Const(0) - o),
                DslUnaryOp::Not => Err(DslError::UnsupportedOperation(
                    "Not not supported in index expression".to_string(),
                )),
            }
        }
        _ => Err(DslError::UnsupportedOperation(format!(
            "Expression type not supported in index expression: {:?}",
            expr
        ))),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn test_compile_simple_add() {
        let source = r#"
            graph<N=10, M=20> main(a: f32[N, M], b: f32[N, M]) -> (c: f32[N, M]) {
                c = a + b
                return c
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
            graph<N=10, M=20> main(x: f32[N, M]) -> (y: f32[1, N, M]) {
                a = x.unsqueeze(0)
                return a
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
            graph<N=10, M=20> relu(x: f32[N, M]) -> (y: f32[N, M]) {
                zero = 0.0
                result = max(x, zero)
                return result
            }

            graph<B=1> main(input: f32[B, 256]) -> (output: f32[B, 256]) {
                output = relu(input)
                return output
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
            graph<N=10> square(x: f32[N]) -> (y: f32[N]) {
                result = x * x
                return result
            }

            graph<N=10> quad(x: f32[N]) -> (y: f32[N]) {
                sq = square(x)
                result = square(sq)
                return result
            }

            graph main(input: f32[10]) -> (output: f32[10]) {
                result = quad(input)
                return result
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
            graph<N=10> add(a: f32[N], b: f32[N]) -> (c: f32[N]) {
                result = a + b
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(matches!(result, Err(DslError::NoMainGraph)));
    }

    #[test]
    fn test_undefined_subgraph_error() {
        let source = r#"
            graph<N=10> main(input: f32[N]) -> (output: f32[N]) {
                result = unknown_func(input)
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(matches!(result, Err(DslError::UndefinedFunction(_))));
    }

    #[test]
    fn test_subgraph_argument_mismatch() {
        let source = r#"
            graph<N=10> add(a: f32[N], b: f32[N]) -> (c: f32[N]) {
                result = a + b
                return result
            }

            graph main(input: f32[10]) -> (output: f32[10]) {
                result = add(input)
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(matches!(
            result,
            Err(DslError::SubgraphArgumentMismatch { .. })
        ));
    }

    #[test]
    fn test_compile_keyword_arg_dim() {
        let source = r#"
            graph<N=10, M=20> main(x: f32[N, M]) -> (y: f32[N]) {
                result = x.sum(dim=1)
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_compile_keyword_arg_axis() {
        let source = r#"
            graph<N=10, M=20> main(x: f32[N, M]) -> (y: f32[N]) {
                result = x.sum(axis=1)
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_compile_positional_arg() {
        let source = r#"
            graph<N=10, M=20> main(x: f32[N, M]) -> (y: f32[N]) {
                result = x.sum(1)
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_tuple_unpacking_multi_output() {
        let source = r#"
            graph<N=10> split_add_mul(x: f32[N]) -> (added: f32[N], multiplied: f32[N]) {
                one = 1.0
                two = 2.0
                a = x + one
                m = x * two
                return a, m
            }

            graph main(input: f32[10]) -> (out1: f32[10], out2: f32[10]) {
                (a, b) = split_add_mul(input)
                return a, b
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
            graph<N=10> two_outputs(x: f32[N]) -> (a: f32[N], b: f32[N]) {
                return x, x
            }

            graph main(input: f32[10]) -> (out: f32[10]) {
                (x, y, z) = two_outputs(input)
                return x
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let result = compile(&module);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("2 outputs") && err_str.contains("3 elements"));
    }

    #[test]
    fn test_shape_var_defaults_registered() {
        let source = r#"
            graph<N=32, M=64> main(x: f32[N, M]) -> (y: f32[N, M]) {
                return x
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.shape_var_defaults().get("N"), Some(&32));
        assert_eq!(graph.shape_var_defaults().get("M"), Some(&64));
    }

    #[test]
    fn test_compile_view_expr() {
        // Test view_expr method for IndexExpr views
        let source = r#"
            graph main(x: f32[12]) -> (y: f32[4, 3]) {
                result = x.view_expr([4, 3], idx0 * 3 + idx1)
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.input_metas().len(), 1);
        assert_eq!(graph.outputs().len(), 1);

        // Verify the output node has an IndexExpr view
        let output = graph.outputs().get("y").expect("output not found");
        assert!(!output.view.is_linear());
        assert!(!output.view.is_contiguous());
    }

    #[test]
    fn test_compile_view_expr_with_modulo() {
        // Test view_expr with modulo operation (circular buffer)
        let source = r#"
            graph main(x: f32[10]) -> (y: f32[10]) {
                result = x.view_expr([10], (idx0 + 5) % 10)
                return result
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let graph = compile(&module).expect("Failed to compile");

        assert_eq!(graph.outputs().len(), 1);
        let output = graph.outputs().get("y").expect("output not found");
        assert!(!output.view.is_linear());
    }
}
