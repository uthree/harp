//! Graph to DSL decompilation

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use harp::graph::ops::{CumulativeOp, ElementwiseOp, GraphOp, ReduceOp};
use harp::graph::shape::Expr;
use harp::graph::{DType, Graph, GraphNode, GraphNodeData};

/// Decompile a Harp Graph to DSL source code
pub fn decompile(graph: &Graph, name: &str) -> String {
    let mut decompiler = Decompiler::new(graph, name);
    decompiler.decompile()
}

struct Decompiler<'a> {
    graph: &'a Graph,
    name: &'a str,
    var_counter: usize,
    node_names: HashMap<*const GraphNodeData, String>,
    used_names: HashSet<String>, // 使用済みの変数名を追跡
    output: String,
}

impl<'a> Decompiler<'a> {
    fn new(graph: &'a Graph, name: &'a str) -> Self {
        Self {
            graph,
            name,
            var_counter: 0,
            node_names: HashMap::new(),
            used_names: HashSet::new(),
            output: String::new(),
        }
    }

    fn decompile(&mut self) -> String {
        // Collect shape variables
        let shape_vars = self.collect_shape_vars();

        // Build header
        let mut header = "graph".to_string();
        if !shape_vars.is_empty() {
            header.push('<');
            header.push_str(&shape_vars.join(", "));
            header.push('>');
        }
        header.push(' ');
        header.push_str(self.name);

        // Build input params
        let input_metas = self.graph.input_metas();
        let input_params: Vec<String> = input_metas
            .iter()
            .map(|meta| {
                let dtype = dtype_to_dsl(&meta.dtype);
                let shape = shape_to_dsl(&meta.shape);
                format!("{}: {}[{}]", meta.name, dtype, shape)
            })
            .collect();

        // Build output params
        let output_metas = self.graph.output_metas();

        // Write signature
        // Use simplified syntax for single output: -> f32[N] instead of -> (output: f32[N])
        if output_metas.len() == 1 {
            let meta = &output_metas[0];
            let dtype = dtype_to_dsl(&meta.dtype);
            let shape = shape_to_dsl(&meta.shape);
            writeln!(
                self.output,
                "{}({}) -> {}[{}] {{",
                header,
                input_params.join(", "),
                dtype,
                shape
            )
            .unwrap();
        } else {
            let output_params: Vec<String> = output_metas
                .iter()
                .map(|meta| {
                    let dtype = dtype_to_dsl(&meta.dtype);
                    let shape = shape_to_dsl(&meta.shape);
                    format!("{}: {}[{}]", meta.name, dtype, shape)
                })
                .collect();
            writeln!(
                self.output,
                "{}({}) -> ({}) {{",
                header,
                input_params.join(", "),
                output_params.join(", ")
            )
            .unwrap();
        }

        // Register input names
        for meta in input_metas {
            // Find the corresponding node by name
            // Input nodes are Buffer nodes with the same name
            self.register_input_name(&meta.name);
        }

        // Process all output nodes (will recursively process dependencies)
        let output_nodes = self.graph.outputs();
        for output_meta in output_metas {
            if let Some(node) = output_nodes.get(&output_meta.name) {
                self.process_node(node);
                // Assign output
                let var_name = self.get_node_name(node);
                if var_name != output_meta.name {
                    writeln!(self.output, "    {} = {}", output_meta.name, var_name).unwrap();
                }
            }
        }

        self.output.push_str("}\n");
        self.output.clone()
    }

    fn collect_shape_vars(&self) -> Vec<String> {
        let mut vars = HashSet::new();

        for meta in self.graph.input_metas() {
            for expr in &meta.shape {
                self.collect_expr_vars(expr, &mut vars);
            }
        }

        for meta in self.graph.output_metas() {
            for expr in &meta.shape {
                self.collect_expr_vars(expr, &mut vars);
            }
        }

        let mut result: Vec<String> = vars.into_iter().collect();
        result.sort();
        result
    }

    fn collect_expr_vars(&self, expr: &Expr, vars: &mut HashSet<String>) {
        match expr {
            Expr::Var(name) => {
                vars.insert(name.clone());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Rem(a, b) => {
                self.collect_expr_vars(a, vars);
                self.collect_expr_vars(b, vars);
            }
            Expr::Const(_) | Expr::Idx(_) => {}
        }
    }

    fn register_input_name(&mut self, name: &str) {
        // Find the input node in the graph
        for meta in self.graph.input_metas() {
            if meta.name == name {
                // We need to find the actual GraphNode for this input
                // For now, just store the name mapping
                // Input nodes will be resolved when processing
                break;
            }
        }
    }

    fn process_node(&mut self, node: &GraphNode) -> String {
        let ptr = node.as_ptr();
        if let Some(name) = self.node_names.get(&ptr) {
            return name.clone();
        }

        // Process dependencies first
        for src in &node.src {
            self.process_node(src);
        }

        // Generate code for this node
        let (name, code) = self.node_to_dsl(node);

        // Store name
        self.node_names.insert(ptr, name.clone());

        // Write code if not empty
        if let Some(code) = code {
            writeln!(self.output, "    let {} = {}", name, code).unwrap();
        }

        name
    }

    fn get_node_name(&self, node: &GraphNode) -> String {
        let ptr = node.as_ptr();
        self.node_names
            .get(&ptr)
            .cloned()
            .unwrap_or_else(|| format!("_unknown_{:p}", ptr))
    }

    fn fresh_var(&mut self) -> String {
        loop {
            let name = format!("v{}", self.var_counter);
            self.var_counter += 1;
            if !self.used_names.contains(&name) {
                self.used_names.insert(name.clone());
                return name;
            }
        }
    }

    /// ノードの名前を取得または生成
    /// ノードに名前が設定されている場合はそれを使用し、重複時はサフィックスを追加
    fn get_or_create_name(&mut self, node: &GraphNode) -> String {
        if let Some(node_name) = node.name() {
            if !self.used_names.contains(node_name) {
                self.used_names.insert(node_name.to_string());
                return node_name.to_string();
            }
            // 名前が重複している場合はサフィックスを追加
            let mut suffix = 1;
            loop {
                let name = format!("{}_{}", node_name, suffix);
                if !self.used_names.contains(&name) {
                    self.used_names.insert(name.clone());
                    return name;
                }
                suffix += 1;
            }
        } else {
            self.fresh_var()
        }
    }

    fn node_to_dsl(&mut self, node: &GraphNode) -> (String, Option<String>) {
        match &node.op {
            GraphOp::Buffer { name } => {
                // Input buffer - use the buffer name directly and register it
                self.used_names.insert(name.clone());

                // Check if the buffer's view is non-contiguous (has view transformations)
                // If so, we need to emit a view operation
                let meta = self.graph.input_metas().iter().find(|m| &m.name == name);

                if let Some(meta) = meta {
                    // Create a contiguous view from the original shape to compare
                    let original_view = harp::graph::shape::View::contiguous(meta.shape.clone());

                    // If the node's view differs from the contiguous original, emit view ops
                    if node.view != original_view {
                        let var_name = self.get_or_create_name(node);
                        // Use original shape for proper view detection
                        let code = self.view_to_dsl_with_shape(name, &meta.shape, &node.view);
                        return (var_name, Some(code));
                    }
                }

                (name.clone(), None)
            }

            GraphOp::Const(lit) => {
                let name = self.get_or_create_name(node);
                let code = match lit {
                    harp::ast::Literal::F32(v) => format!("{}", v),
                    harp::ast::Literal::Int(v) => format!("{}", v),
                    harp::ast::Literal::Bool(v) => format!("{}", v),
                };
                (name, Some(code))
            }

            GraphOp::Elementwise { op } => {
                let name = self.get_or_create_name(node);
                let code = self.elementwise_to_dsl(op, &node.src);
                (name, Some(code))
            }

            GraphOp::Reduce { op, axis, .. } => {
                let name = self.get_or_create_name(node);
                let src_name = self.get_node_name(&node.src[0]);
                let method = reduce_op_to_method(op);
                let code = format!("{}.{}({})", src_name, method, axis);
                (name, Some(code))
            }

            GraphOp::Cumulative { op, axis, .. } => {
                let name = self.get_or_create_name(node);
                let src_name = self.get_node_name(&node.src[0]);
                let method = cumulative_op_to_method(op);
                let code = format!("{}.{}({})", src_name, method, axis);
                (name, Some(code))
            }

            GraphOp::View(view) => {
                let name = self.get_or_create_name(node);
                let src_name = self.get_node_name(&node.src[0]);
                // Try to identify the view operation
                let code = self.view_to_dsl(&src_name, &node.src[0], view);
                (name, Some(code))
            }

            GraphOp::FusedElementwise { expr } => {
                let name = self.get_or_create_name(node);
                let inputs: Vec<String> = node.src.iter().map(|n| self.get_node_name(n)).collect();
                let expr_str = ast_expr_to_dsl(expr, &inputs);
                let code = format!("fused({}) {{ {} }}", inputs.join(", "), expr_str);
                (name, Some(code))
            }

            GraphOp::FusedElementwiseReduce {
                expr,
                reduce_op,
                axes,
                ..
            } => {
                let name = self.get_or_create_name(node);
                let inputs: Vec<String> = node.src.iter().map(|n| self.get_node_name(n)).collect();
                let expr_str = ast_expr_to_dsl(expr, &inputs);
                let op_name = reduce_op_to_name(reduce_op);
                let axis = axes.first().copied().unwrap_or(0);
                let code = format!(
                    "fused_reduce({}, axis={}, op={}) {{ {} }}",
                    inputs.join(", "),
                    axis,
                    op_name,
                    expr_str
                );
                (name, Some(code))
            }

            GraphOp::FusedElementwiseCumulative {
                expr,
                cumulative_op,
                axis,
                ..
            } => {
                let name = self.get_or_create_name(node);
                let inputs: Vec<String> = node.src.iter().map(|n| self.get_node_name(n)).collect();
                let expr_str = ast_expr_to_dsl(expr, &inputs);
                let op_name = cumulative_op_to_name(cumulative_op);
                let code = format!(
                    "fused_cumulative({}, axis={}, op={}) {{ {} }}",
                    inputs.join(", "),
                    axis,
                    op_name,
                    expr_str
                );
                (name, Some(code))
            }

            GraphOp::Concat { axis } => {
                let name = self.get_or_create_name(node);
                let inputs: Vec<String> = node.src.iter().map(|n| self.get_node_name(n)).collect();
                let code = format!("concat([{}], {})", inputs.join(", "), axis);
                (name, Some(code))
            }

            GraphOp::Pad { padding, value } => {
                let name = self.get_or_create_name(node);
                let src_name = self.get_node_name(&node.src[0]);
                let padding_str: Vec<String> = padding
                    .iter()
                    .map(|(before, after)| format!("({}, {})", before, after))
                    .collect();
                let code = format!("{}.pad([{}], {})", src_name, padding_str.join(", "), value);
                (name, Some(code))
            }

            GraphOp::Slice { ranges } => {
                let name = self.get_or_create_name(node);
                let src_name = self.get_node_name(&node.src[0]);
                let ranges_str: Vec<String> = ranges
                    .iter()
                    .map(|(start, end)| format!("({}, {})", start, end))
                    .collect();
                let code = format!("{}.slice([{}])", src_name, ranges_str.join(", "));
                (name, Some(code))
            }

            GraphOp::Cast { target_dtype } => {
                let name = self.get_or_create_name(node);
                let src_name = self.get_node_name(&node.src[0]);
                let dtype = dtype_to_dsl(target_dtype);
                let code = format!("{}.cast({})", src_name, dtype);
                (name, Some(code))
            }

            // Unsupported ops get a comment
            _ => {
                let name = self.get_or_create_name(node);
                let code = format!("/* unsupported: {:?} */", std::mem::discriminant(&node.op));
                (name, Some(code))
            }
        }
    }

    fn elementwise_to_dsl(&self, op: &ElementwiseOp, src: &[GraphNode]) -> String {
        match op {
            ElementwiseOp::Add => {
                let a = self.get_node_name(&src[0]);
                let b = self.get_node_name(&src[1]);
                format!("{} + {}", a, b)
            }
            ElementwiseOp::Mul => {
                let a = self.get_node_name(&src[0]);
                let b = self.get_node_name(&src[1]);
                format!("{} * {}", a, b)
            }
            ElementwiseOp::Max => {
                let a = self.get_node_name(&src[0]);
                let b = self.get_node_name(&src[1]);
                format!("max({}, {})", a, b)
            }
            ElementwiseOp::Neg => {
                let a = self.get_node_name(&src[0]);
                format!("-{}", a)
            }
            ElementwiseOp::Recip => {
                let a = self.get_node_name(&src[0]);
                format!("{}.recip()", a)
            }
            ElementwiseOp::Log2 => {
                let a = self.get_node_name(&src[0]);
                format!("{}.log2()", a)
            }
            ElementwiseOp::Exp2 => {
                let a = self.get_node_name(&src[0]);
                format!("{}.exp2()", a)
            }
            ElementwiseOp::Sqrt => {
                let a = self.get_node_name(&src[0]);
                format!("{}.sqrt()", a)
            }
            ElementwiseOp::Sin => {
                let a = self.get_node_name(&src[0]);
                format!("{}.sin()", a)
            }
            ElementwiseOp::Idiv => {
                let a = self.get_node_name(&src[0]);
                let b = self.get_node_name(&src[1]);
                format!("{} / {}", a, b)
            }
            ElementwiseOp::Rem => {
                let a = self.get_node_name(&src[0]);
                let b = self.get_node_name(&src[1]);
                format!("{} % {}", a, b)
            }
        }
    }

    fn view_to_dsl(
        &self,
        src_name: &str,
        src_node: &GraphNode,
        view: &harp::graph::shape::View,
    ) -> String {
        self.view_to_dsl_with_shape(src_name, src_node.view.shape(), view)
    }

    fn view_to_dsl_with_shape(
        &self,
        src_name: &str,
        src_shape: &[Expr],
        view: &harp::graph::shape::View,
    ) -> String {
        use harp::graph::shape::View;
        let new_shape = view.shape();

        // Handle IndexExpr view
        if let View::IndexExpr { index_expr, .. } = view {
            let shape_str = shape_to_dsl(new_shape);
            let expr_str = expr_to_dsl(index_expr);
            return format!("{}.view_expr([{}], {})", src_name, shape_str, expr_str);
        }

        let View::Linear {
            strides: new_strides,
            offset: new_offset,
            ..
        } = view
        else {
            unreachable!()
        };

        // Check for unsqueeze: new dimension added with stride 0
        if new_shape.len() == src_shape.len() + 1 {
            for (i, (shape_expr, stride_expr)) in
                new_shape.iter().zip(new_strides.iter()).enumerate()
            {
                if *stride_expr == Expr::Const(0) && *shape_expr == Expr::Const(1) {
                    return format!("{}.unsqueeze({})", src_name, i);
                }
            }
        }

        // Check for squeeze: dimension removed
        if new_shape.len() == src_shape.len() - 1 {
            for (i, shape_expr) in src_shape.iter().enumerate() {
                if *shape_expr == Expr::Const(1) {
                    return format!("{}.squeeze({})", src_name, i);
                }
            }
        }

        // Check for permute: same shape, different strides order
        if new_shape.len() == src_shape.len() {
            // Try to detect permutation
            // This is a simplified check
        }

        // Check for repeat: some strides are 0 with non-1 shape (broadcast expansion)
        let repeat_axes: Vec<usize> = new_strides
            .iter()
            .enumerate()
            .filter_map(|(i, stride)| {
                if *stride == Expr::Const(0) && new_shape[i] != Expr::Const(1) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if !repeat_axes.is_empty() {
            let mut result = src_name.to_string();
            for axis in repeat_axes {
                let times = expr_to_dsl(&new_shape[axis]);
                result = format!("{}.repeat({}, {})", result, axis, times);
            }
            return result;
        }

        // Default: generic view operation
        let shape_str = shape_to_dsl(new_shape);
        let strides_str = shape_to_dsl(new_strides);
        format!(
            "{}.view([{}], [{}], {})",
            src_name,
            shape_str,
            strides_str,
            expr_to_dsl(new_offset)
        )
    }
}

// Helper functions

fn dtype_to_dsl(dtype: &DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::Bool => "bool",
        DType::Unknown => "unknown",
    }
}

fn shape_to_dsl(shape: &[Expr]) -> String {
    shape.iter().map(expr_to_dsl).collect::<Vec<_>>().join(", ")
}

fn expr_to_dsl(expr: &Expr) -> String {
    match expr {
        Expr::Const(v) => v.to_string(),
        Expr::Var(name) => name.clone(),
        Expr::Idx(i) => format!("idx{}", i),
        Expr::Add(a, b) => format!("({} + {})", expr_to_dsl(a), expr_to_dsl(b)),
        Expr::Sub(a, b) => format!("({} - {})", expr_to_dsl(a), expr_to_dsl(b)),
        Expr::Mul(a, b) => format!("({} * {})", expr_to_dsl(a), expr_to_dsl(b)),
        Expr::Div(a, b) => format!("({} / {})", expr_to_dsl(a), expr_to_dsl(b)),
        Expr::Rem(a, b) => format!("({} % {})", expr_to_dsl(a), expr_to_dsl(b)),
    }
}

fn reduce_op_to_method(op: &ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Prod => "prod",
        ReduceOp::Max => "max",
    }
}

fn reduce_op_to_name(op: &ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Prod => "prod",
        ReduceOp::Max => "max",
    }
}

fn cumulative_op_to_method(op: &CumulativeOp) -> &'static str {
    match op {
        CumulativeOp::Sum => "cumsum",
        CumulativeOp::Prod => "cumprod",
    }
}

fn cumulative_op_to_name(op: &CumulativeOp) -> &'static str {
    match op {
        CumulativeOp::Sum => "sum",
        CumulativeOp::Prod => "prod",
    }
}

fn ast_expr_to_dsl(expr: &harp::ast::AstNode, inputs: &[String]) -> String {
    use harp::ast::{AstNode, Literal};

    match expr {
        AstNode::Wildcard(name) => {
            // Convert wildcard index to input name
            if let Ok(idx) = name.parse::<usize>()
                && idx < inputs.len()
            {
                return inputs[idx].clone();
            }
            format!("${}", name)
        }
        AstNode::Const(lit) => match lit {
            Literal::F32(v) => format!("{}", v),
            Literal::Int(v) => format!("{}", v),
            Literal::Bool(v) => format!("{}", v),
        },
        AstNode::Add(a, b) => {
            // Check for subtraction pattern: Add(a, Mul(Const(-1), b))
            if let AstNode::Mul(left, right) = b.as_ref()
                && let AstNode::Const(Literal::F32(v)) = left.as_ref()
                && *v == -1.0
            {
                return format!(
                    "({} - {})",
                    ast_expr_to_dsl(a, inputs),
                    ast_expr_to_dsl(right, inputs)
                );
            }
            format!(
                "({} + {})",
                ast_expr_to_dsl(a, inputs),
                ast_expr_to_dsl(b, inputs)
            )
        }
        AstNode::Mul(a, b) => {
            // Check for negation pattern: Mul(Const(-1), x)
            if let AstNode::Const(Literal::F32(v)) = a.as_ref()
                && *v == -1.0
            {
                return format!("(-{})", ast_expr_to_dsl(b, inputs));
            }
            // Check for division pattern: Mul(a, Recip(b))
            if let AstNode::Recip(inner) = b.as_ref() {
                return format!(
                    "({} / {})",
                    ast_expr_to_dsl(a, inputs),
                    ast_expr_to_dsl(inner, inputs)
                );
            }
            format!(
                "({} * {})",
                ast_expr_to_dsl(a, inputs),
                ast_expr_to_dsl(b, inputs)
            )
        }
        AstNode::Rem(a, b) => format!(
            "({} % {})",
            ast_expr_to_dsl(a, inputs),
            ast_expr_to_dsl(b, inputs)
        ),
        AstNode::Recip(a) => format!("(1.0 / {})", ast_expr_to_dsl(a, inputs)),
        AstNode::Max(a, b) => format!(
            "max({}, {})",
            ast_expr_to_dsl(a, inputs),
            ast_expr_to_dsl(b, inputs)
        ),
        AstNode::Sqrt(a) => format!("sqrt({})", ast_expr_to_dsl(a, inputs)),
        AstNode::Log2(a) => format!("log2({})", ast_expr_to_dsl(a, inputs)),
        AstNode::Exp2(a) => format!("exp2({})", ast_expr_to_dsl(a, inputs)),
        AstNode::Sin(a) => format!("sin({})", ast_expr_to_dsl(a, inputs)),
        _ => "/* unsupported ast expr */".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompile_simple() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4, 8]);
        let b = graph.input("b", DType::F32, vec![4, 8]);
        let c = &a + &b;
        graph.output("c", c);

        let dsl = decompile(&graph, "add");
        println!("{}", dsl);

        assert!(dsl.contains("graph add"));
        assert!(dsl.contains("a: f32[4, 8]"));
        assert!(dsl.contains("b: f32[4, 8]"));
        // Single output uses simplified syntax: -> f32[4, 8]
        assert!(dsl.contains("-> f32[4, 8]"));
    }

    #[test]
    fn test_decompile_with_shape_vars() {
        let mut graph = Graph::new();
        let shape = vec![
            harp::graph::shape::Expr::Var("N".to_string()),
            harp::graph::shape::Expr::Var("M".to_string()),
        ];
        let a = graph.input("a", DType::F32, shape.clone());
        let b = graph.input("b", DType::F32, shape);
        let c = &a + &b;
        graph.output("c", c);

        let dsl = decompile(&graph, "add");
        println!("{}", dsl);

        assert!(dsl.contains("graph<"));
        assert!(dsl.contains("M") && dsl.contains("N"));
    }

    #[test]
    fn test_decompile_input_with_view() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4, 8]);

        // Create a view with permute (transpose)
        let a_transposed = a.view(a.view.clone().permute(vec![1, 0]));

        // Use the transposed input in computation
        let b = graph.input("b", DType::F32, vec![8, 4]);
        let c = &a_transposed + &b;
        graph.output("c", c);

        let dsl = decompile(&graph, "transpose_add");
        println!("=== test_decompile_input_with_view ===\n{}", dsl);

        // Should contain a view operation for the transposed input
        assert!(dsl.contains(".view(") || dsl.contains(".permute("));
    }

    #[test]
    fn test_decompile_input_with_repeat() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![1, 8]);

        // Create a view with repeat (broadcast axis 0 from 1 to 4)
        let a_repeated = a.view(a.view.clone().repeat(0, 4));

        // Use the repeated input in computation
        let b = graph.input("b", DType::F32, vec![4, 8]);
        let c = &a_repeated + &b;
        graph.output("c", c);

        let dsl = decompile(&graph, "repeat_add");
        println!("=== test_decompile_input_with_repeat ===\n{}", dsl);

        // Should contain a repeat operation for the broadcasted input
        assert!(dsl.contains(".repeat(0, 4)"));
    }
}
