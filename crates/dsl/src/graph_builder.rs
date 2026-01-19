//! DSL AST to GraphNode conversion

use std::collections::HashMap;

use eclat::ast::{AstNode as AstNodeLow, DType as AstDType, Literal};
use eclat::graph::shape::{Expr, PadValue};
use eclat::graph::{GraphNode, GraphOp, named_input, scalar};

use crate::ast::*;
use crate::errors::{DslError, DslResult};

/// Converts DSL AST to GraphNode computation graphs
pub struct GraphBuilder {
    /// Dynamic dimension values (name -> value)
    dynamic_dims: HashMap<String, i64>,
}

impl GraphBuilder {
    /// Create a new GraphBuilder
    pub fn new() -> Self {
        GraphBuilder {
            dynamic_dims: HashMap::new(),
        }
    }

    /// Create a new GraphBuilder with dynamic dimension definitions
    pub fn with_dynamic_dims(dims: HashMap<String, i64>) -> Self {
        GraphBuilder { dynamic_dims: dims }
    }

    /// Add a dynamic dimension value
    pub fn define_dim(&mut self, name: &str, value: i64) {
        self.dynamic_dims.insert(name.to_string(), value);
    }

    /// Build a graph from a single GraphDef
    pub fn build_graph(&self, graph_def: &GraphDef) -> DslResult<BuiltGraph> {
        let mut ctx = BuildContext::new(&self.dynamic_dims);

        // Create input nodes for parameters
        for param in &graph_def.params {
            let shape = self.resolve_shape(&param.type_spec.shape)?;
            let dtype = convert_dtype(&param.type_spec.dtype);
            let node = named_input(&param.name, shape, dtype);
            ctx.bind(&param.name, node);
        }

        // Process statements
        for stmt in &graph_def.body {
            match stmt {
                Statement::Let { name, value } => {
                    let node = self.build_expr(&ctx, value)?;
                    ctx.bind(name, node);
                }
            }
        }

        // Build return expression
        let output = self.build_expr(&ctx, &graph_def.return_expr)?;

        Ok(BuiltGraph {
            name: graph_def.name.clone(),
            inputs: ctx.inputs.clone(),
            output,
        })
    }

    /// Build all graphs from a DslProgram
    pub fn build_program(&self, program: &DslProgram) -> DslResult<Vec<BuiltGraph>> {
        program
            .graphs
            .iter()
            .map(|g| self.build_graph(g))
            .collect()
    }

    /// Resolve a shape to concrete Expr values
    fn resolve_shape(&self, shape: &[ShapeDim]) -> DslResult<Vec<Expr>> {
        shape
            .iter()
            .map(|dim| match dim {
                ShapeDim::Static(n) => Ok(Expr::Const(*n)),
                ShapeDim::Dynamic(name) => {
                    self.dynamic_dims
                        .get(name)
                        .map(|&v| Expr::Const(v))
                        .ok_or_else(|| DslError::UnresolvedDynamicDim(name.clone()))
                }
            })
            .collect()
    }

    /// Resolve shape to i64 values (for operations that need concrete values)
    fn resolve_shape_i64(&self, shape: &[ShapeDim]) -> DslResult<Vec<i64>> {
        shape
            .iter()
            .map(|dim| match dim {
                ShapeDim::Static(n) => Ok(*n),
                ShapeDim::Dynamic(name) => self
                    .dynamic_dims
                    .get(name)
                    .copied()
                    .ok_or_else(|| DslError::UnresolvedDynamicDim(name.clone())),
            })
            .collect()
    }

    /// Build expression to GraphNode
    fn build_expr(&self, ctx: &BuildContext, expr: &DslExpr) -> DslResult<GraphNode> {
        match expr {
            DslExpr::Var(name) => ctx
                .lookup(name)
                .ok_or_else(|| DslError::UndefinedVariable(name.clone())),

            DslExpr::Literal(value) => {
                // Create a scalar constant using MapReduce with a constant value
                let scalar_node = scalar(AstDType::F32);
                Ok(GraphNode::new(
                    vec![scalar_node.clone()],
                    scalar_node.view().clone(),
                    GraphOp::MapReduce {
                        map: AstNodeLow::Const(Literal::F32(*value as f32)),
                        reduce: None,
                    },
                    AstDType::F32,
                    None,
                ))
            }

            DslExpr::BinaryOp { op, lhs, rhs } => {
                let lhs_node = self.build_expr(ctx, lhs)?;
                let rhs_node = self.build_expr(ctx, rhs)?;
                Ok(apply_binary_op(*op, lhs_node, rhs_node))
            }

            DslExpr::UnaryOp { op, operand } => {
                let operand_node = self.build_expr(ctx, operand)?;
                Ok(apply_unary_op(*op, operand_node))
            }

            DslExpr::FuncCall { name, args } => self.build_func_call(ctx, name, args),
        }
    }

    /// Build a function call
    fn build_func_call(
        &self,
        ctx: &BuildContext,
        name: &str,
        args: &[FuncArg],
    ) -> DslResult<GraphNode> {
        match name {
            // Math functions (unary)
            "sqrt" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.sqrt())
            }
            "exp" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.exp())
            }
            "log" | "ln" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.ln())
            }
            "sin" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.sin())
            }
            "cos" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.cos())
            }
            "floor" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.floor())
            }
            "abs" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.abs())
            }
            "recip" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                Ok(arg.recip())
            }

            // Reduction functions
            "sum" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axis = self.get_named_int(args, "axis")?;
                Ok(arg.sum(axis))
            }
            "max" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axis = self.get_named_int(args, "axis")?;
                Ok(arg.max(axis))
            }
            "min" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axis = self.get_named_int(args, "axis")?;
                Ok(arg.min(axis))
            }
            "prod" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axis = self.get_named_int(args, "axis")?;
                Ok(arg.prod(axis))
            }

            // Shape transformations
            "permute" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let perm = self.get_positional_shape_i64(args, 1)?;
                let perm_usize: Vec<usize> = perm.iter().map(|&x| x as usize).collect();
                Ok(arg.permute(&perm_usize))
            }
            "reshape" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let shape = self.get_positional_shape(args, 1)?;
                Ok(arg.reshape(shape))
            }
            "unsqueeze" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axis = self.get_named_int(args, "axis")?;
                Ok(arg.unsqueeze(axis))
            }
            "squeeze" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axis = self.get_named_int(args, "axis")?;
                Ok(arg.squeeze(axis))
            }
            "expand" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let target_shape = self.get_positional_shape(args, 1)?;

                // Expand the tensor to the target shape by expanding each dimension
                let _current_shape = arg.shape();
                let mut result = arg;

                // Handle dimension mismatch by unsqueezing
                while result.shape().len() < target_shape.len() {
                    result = result.unsqueeze(0);
                }

                // Expand each dimension that is 1 to the target size
                for (axis, target_dim) in target_shape.iter().enumerate() {
                    if axis < result.shape().len() {
                        let current_dim = &result.shape()[axis];
                        if *current_dim == Expr::Const(1) && current_dim != target_dim {
                            result = result.expand(axis, target_dim.clone());
                        }
                    }
                }

                Ok(result)
            }
            "flip" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axis = self.get_named_int(args, "axis")?;
                Ok(arg.flip(axis))
            }

            // Conditional
            "where" => {
                let cond = self.get_positional_expr(ctx, args, 0)?;
                let then_val = self.get_positional_expr(ctx, args, 1)?;
                let else_val = self.get_positional_expr(ctx, args, 2)?;
                Ok(cond.where_cond(&then_val, &else_val))
            }

            // Type cast
            "cast" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let dtype = self.get_positional_dtype(args, 1)?;
                Ok(arg.cast(dtype))
            }

            // Unfold
            "unfold" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let axes = self.get_named_shape_i64(args, "axes")?;
                let sizes = self.get_named_shape(args, "sizes")?;
                let strides = self.get_named_shape(args, "strides")?;

                let axes_usize: Vec<usize> = axes.iter().map(|&x| x as usize).collect();

                // Default dilations to 1
                let dilations = if let Ok(d) = self.get_named_shape(args, "dilations") {
                    d
                } else {
                    vec![Expr::Const(1); axes.len()]
                };

                Ok(arg.unfold(&axes_usize, &sizes, &strides, &dilations))
            }

            // Pad
            "pad" => {
                let arg = self.get_positional_expr(ctx, args, 0)?;
                let padding = self.get_positional_padding(args, 1)?;
                // Use PadValue::Zero by default
                Ok(arg.pad(&padding, PadValue::Zero))
            }

            _ => Err(DslError::UnknownFunction(name.to_string())),
        }
    }

    /// Get a positional expression argument
    fn get_positional_expr(
        &self,
        ctx: &BuildContext,
        args: &[FuncArg],
        index: usize,
    ) -> DslResult<GraphNode> {
        match args.get(index) {
            Some(FuncArg::Positional(expr)) => self.build_expr(ctx, expr),
            Some(FuncArg::Named { value, .. }) => self.build_expr(ctx, value),
            _ => Err(DslError::InvalidArgCount {
                func: "function".to_string(),
                expected: index + 1,
                got: args.len(),
            }),
        }
    }

    /// Get a named integer argument (for axis, etc.)
    fn get_named_int(&self, args: &[FuncArg], name: &str) -> DslResult<usize> {
        for arg in args {
            if let FuncArg::Named {
                name: arg_name,
                value,
            } = arg
            {
                if arg_name == name {
                    if let DslExpr::Literal(n) = value {
                        return Ok(*n as usize);
                    }
                }
            }
        }
        Err(DslError::InvalidArgType {
            func: "function".to_string(),
            arg: name.to_string(),
            expected: "integer".to_string(),
            got: "missing or invalid".to_string(),
        })
    }

    /// Get a positional shape argument as Vec<Expr>
    fn get_positional_shape(&self, args: &[FuncArg], index: usize) -> DslResult<Vec<Expr>> {
        match args.get(index) {
            Some(FuncArg::Shape(dims)) => self.resolve_shape(dims),
            Some(FuncArg::NamedShape { shape, .. }) => self.resolve_shape(shape),
            _ => Err(DslError::InvalidArgType {
                func: "function".to_string(),
                arg: format!("arg{}", index),
                expected: "shape".to_string(),
                got: "other".to_string(),
            }),
        }
    }

    /// Get a positional shape argument as Vec<i64>
    fn get_positional_shape_i64(&self, args: &[FuncArg], index: usize) -> DslResult<Vec<i64>> {
        match args.get(index) {
            Some(FuncArg::Shape(dims)) => self.resolve_shape_i64(dims),
            Some(FuncArg::NamedShape { shape, .. }) => self.resolve_shape_i64(shape),
            _ => Err(DslError::InvalidArgType {
                func: "function".to_string(),
                arg: format!("arg{}", index),
                expected: "shape".to_string(),
                got: "other".to_string(),
            }),
        }
    }

    /// Get a named shape argument as Vec<Expr>
    fn get_named_shape(&self, args: &[FuncArg], name: &str) -> DslResult<Vec<Expr>> {
        for arg in args {
            if let FuncArg::NamedShape {
                name: arg_name,
                shape,
            } = arg
            {
                if arg_name == name {
                    return self.resolve_shape(shape);
                }
            }
        }
        Err(DslError::InvalidArgType {
            func: "function".to_string(),
            arg: name.to_string(),
            expected: "shape".to_string(),
            got: "missing".to_string(),
        })
    }

    /// Get a named shape argument as Vec<i64>
    fn get_named_shape_i64(&self, args: &[FuncArg], name: &str) -> DslResult<Vec<i64>> {
        for arg in args {
            if let FuncArg::NamedShape {
                name: arg_name,
                shape,
            } = arg
            {
                if arg_name == name {
                    return self.resolve_shape_i64(shape);
                }
            }
        }
        Err(DslError::InvalidArgType {
            func: "function".to_string(),
            arg: name.to_string(),
            expected: "shape".to_string(),
            got: "missing".to_string(),
        })
    }

    /// Get a positional dtype argument
    fn get_positional_dtype(&self, args: &[FuncArg], index: usize) -> DslResult<AstDType> {
        match args.get(index) {
            Some(FuncArg::Positional(DslExpr::Var(name))) => match name.as_str() {
                "f16" => Ok(AstDType::F16),
                "bf16" => Ok(AstDType::BF16),
                "f32" => Ok(AstDType::F32),
                "f64" => Ok(AstDType::F64),
                "i32" => Ok(AstDType::I32),
                "i64" => Ok(AstDType::I64),
                "u32" => Ok(AstDType::U32),
                "u64" => Ok(AstDType::U64),
                "bool" => Ok(AstDType::Bool),
                other => Err(DslError::UnknownDType(other.to_string())),
            },
            _ => Err(DslError::InvalidArgType {
                func: "cast".to_string(),
                arg: format!("arg{}", index),
                expected: "dtype".to_string(),
                got: "other".to_string(),
            }),
        }
    }

    /// Get a positional padding argument: [(before, after), ...]
    fn get_positional_padding(
        &self,
        args: &[FuncArg],
        index: usize,
    ) -> DslResult<Vec<(Expr, Expr)>> {
        // Padding is expected as a shape-like argument: [0, 0, 1, 1, ...]
        // where each pair is (before, after) for each dimension
        match args.get(index) {
            Some(FuncArg::Shape(dims)) => {
                let values = self.resolve_shape(dims)?;
                if values.len() % 2 != 0 {
                    return Err(DslError::InvalidArgType {
                        func: "pad".to_string(),
                        arg: "padding".to_string(),
                        expected: "even number of values".to_string(),
                        got: format!("{} values", values.len()),
                    });
                }
                Ok(values
                    .chunks(2)
                    .map(|c| (c[0].clone(), c[1].clone()))
                    .collect())
            }
            _ => Err(DslError::InvalidArgType {
                func: "pad".to_string(),
                arg: format!("arg{}", index),
                expected: "padding shape".to_string(),
                got: "other".to_string(),
            }),
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for building expressions
struct BuildContext<'a> {
    /// Variable bindings (name -> GraphNode)
    bindings: HashMap<String, GraphNode>,
    /// Input parameters in order
    inputs: Vec<(String, GraphNode)>,
    /// Dynamic dimension values
    #[allow(dead_code)]
    dynamic_dims: &'a HashMap<String, i64>,
}

impl<'a> BuildContext<'a> {
    fn new(dynamic_dims: &'a HashMap<String, i64>) -> Self {
        BuildContext {
            bindings: HashMap::new(),
            inputs: Vec::new(),
            dynamic_dims,
        }
    }

    fn bind(&mut self, name: &str, node: GraphNode) {
        // If this is an input (external node), add to inputs list
        if node.is_external() && !self.inputs.iter().any(|(n, _)| n == name) {
            self.inputs.push((name.to_string(), node.clone()));
        }
        self.bindings.insert(name.to_string(), node);
    }

    fn lookup(&self, name: &str) -> Option<GraphNode> {
        self.bindings.get(name).cloned()
    }
}

/// Result of building a graph
#[derive(Debug, Clone)]
pub struct BuiltGraph {
    /// Graph name
    pub name: String,
    /// Input parameters (name, node)
    pub inputs: Vec<(String, GraphNode)>,
    /// Output node
    pub output: GraphNode,
}

/// Convert DSL DType to AST DType
fn convert_dtype(dtype: &DType) -> AstDType {
    match dtype {
        DType::F16 => AstDType::F16,
        DType::BF16 => AstDType::BF16,
        DType::F32 => AstDType::F32,
        DType::F64 => AstDType::F64,
        DType::I32 => AstDType::I32,
        DType::I64 => AstDType::I64,
        DType::U32 => AstDType::U32,
        DType::U64 => AstDType::U64,
        DType::Bool => AstDType::Bool,
    }
}

/// Apply a binary operator to two GraphNodes
fn apply_binary_op(op: BinOp, lhs: GraphNode, rhs: GraphNode) -> GraphNode {
    match op {
        BinOp::Add => &lhs + &rhs,
        BinOp::Sub => &lhs - &rhs,
        BinOp::Mul => &lhs * &rhs,
        BinOp::Div => &lhs / &rhs,
        BinOp::Lt => lhs.lt(&rhs),
        BinOp::Gt => lhs.gt(&rhs),
        BinOp::Le => lhs.le(&rhs),
        BinOp::Ge => lhs.ge(&rhs),
        BinOp::Eq => lhs.eq_node(&rhs),
        BinOp::Ne => lhs.ne_node(&rhs),
    }
}

/// Apply a unary operator to a GraphNode
fn apply_unary_op(op: UnaryOp, operand: GraphNode) -> GraphNode {
    match op {
        UnaryOp::Neg => -operand,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_program;

    #[test]
    fn test_build_simple_add() {
        let source = r#"
program {
    graph add(a: f32[16, 256], b: f32[16, 256]) -> f32[16, 256] {
        return a + b;
    }
}
"#;
        let program = parse_program(source).unwrap();
        let builder = GraphBuilder::new();
        let graphs = builder.build_program(&program).unwrap();

        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].name, "add");
        assert_eq!(graphs[0].inputs.len(), 2);
    }

    #[test]
    fn test_build_with_let() {
        let source = r#"
program {
    graph test(x: f32[10]) -> f32[10] {
        let y = sqrt(x);
        return y;
    }
}
"#;
        let program = parse_program(source).unwrap();
        let builder = GraphBuilder::new();
        let graphs = builder.build_program(&program).unwrap();

        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].name, "test");
    }

    #[test]
    fn test_build_with_dynamic_dims() {
        let source = r#"
program {
    graph test(x: f32[batch, 64]) -> f32[batch, 64] {
        return sqrt(x);
    }
}
"#;
        let program = parse_program(source).unwrap();
        let mut dims = HashMap::new();
        dims.insert("batch".to_string(), 32);
        let builder = GraphBuilder::with_dynamic_dims(dims);
        let graphs = builder.build_program(&program).unwrap();

        assert_eq!(graphs.len(), 1);
    }

    #[test]
    fn test_build_softmax() {
        let source = r#"
program {
    graph softmax(x: f32[32, 64]) -> f32[32, 64] {
        let x_max = max(x, axis=1);
        let x_shifted = x - expand(x_max, [32, 64]);
        let exp_x = exp(x_shifted);
        let sum_exp = sum(exp_x, axis=1);
        return exp_x / expand(sum_exp, [32, 64]);
    }
}
"#;
        let program = parse_program(source).unwrap();
        let builder = GraphBuilder::new();
        let graphs = builder.build_program(&program).unwrap();

        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].name, "softmax");
    }
}
