//! C-like language renderer trait and common implementations
//!
//! This module provides a trait for rendering AST to C-like languages (C, CUDA, Metal, OpenCL, etc.)
//! and provides common implementations that can be shared across these languages.

use crate::ast::{AstNode, ConstLiteral, DType, Scope, VariableDecl};
use std::fmt::Write;

/// Configuration for memory management in C-like languages
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Allocation function (e.g., "malloc", "cudaMalloc")
    pub alloc_fn: String,
    /// Deallocation function (e.g., "free", "cudaFree")
    pub dealloc_fn: String,
    /// Whether to cast the result of allocation
    pub needs_cast: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            alloc_fn: "malloc".to_string(),
            dealloc_fn: "free".to_string(),
            needs_cast: true,
        }
    }
}

/// Trait for rendering AST to C-like languages
///
/// This trait provides common rendering logic for C-like languages (C, CUDA, Metal, OpenCL, etc.)
/// and allows language-specific customization through associated types and methods.
pub trait CLikeRenderer {
    /// Get the indentation level
    fn indent_level(&self) -> usize;

    /// Set the indentation level
    fn set_indent_level(&mut self, level: usize);

    /// Get memory configuration
    fn memory_config(&self) -> &MemoryConfig;

    // ========================================================================
    // Language-specific customization points
    // ========================================================================

    /// Render includes/imports for the language
    fn render_includes(&self) -> String;

    /// Render a scalar data type name (e.g., "float", "size_t")
    fn render_scalar_dtype(&self, dtype: &DType) -> String;

    /// Render a constant literal
    fn render_const(&self, c: &ConstLiteral) -> String;

    /// Render a mathematical function call (allows language-specific function names)
    fn render_math_function(&mut self, name: &str, args: Vec<String>) -> String {
        // Default implementation for most C-like languages
        format!("{}({})", name, args.join(", "))
    }

    /// Render a barrier synchronization
    fn render_barrier(&self) -> String {
        "/* BARRIER */".to_string()
    }

    // ========================================================================
    // Common rendering logic (with default implementations)
    // ========================================================================

    /// Get the precedence level of an operator
    fn precedence(&self, node: &AstNode) -> u8 {
        match node {
            // Highest precedence: atoms and function calls
            AstNode::Const(_) | AstNode::Var(_) | AstNode::CallFunction { .. } => 100,

            // Unary operators
            AstNode::Neg(_)
            | AstNode::Recip(_)
            | AstNode::Sin(_)
            | AstNode::Sqrt(_)
            | AstNode::Log2(_)
            | AstNode::Exp2(_)
            | AstNode::Load { .. }
            | AstNode::Cast { .. } => 90,

            // Multiplicative operators
            AstNode::Mul(_, _) | AstNode::Rem(_, _) => 80,

            // Additive operators
            AstNode::Add(_, _) => 70,

            // Shift operators
            AstNode::Shl(_, _) | AstNode::Shr(_, _) => 60,

            // Comparison operators
            AstNode::LessThan(_, _) | AstNode::Eq(_, _) => 55,

            // Bitwise AND
            AstNode::BitAnd(_, _) => 50,

            // Bitwise XOR
            AstNode::BitXor(_, _) => 40,

            // Bitwise OR
            AstNode::BitOr(_, _) => 30,

            // Ternary/Select operator
            AstNode::Select { .. } => 20,

            // Everything else (statements, etc.)
            _ => 0,
        }
    }

    /// Render a node with parentheses if needed based on precedence
    fn render_with_parens(
        &mut self,
        node: &AstNode,
        parent_precedence: u8,
        is_rhs: bool,
    ) -> String {
        let node_precedence = self.precedence(node);
        let needs_parens = if is_rhs {
            // Right side needs parens if precedence is lower or equal (for left-associative ops)
            node_precedence <= parent_precedence && parent_precedence > 0
        } else {
            // Left side needs parens only if precedence is strictly lower
            node_precedence < parent_precedence && parent_precedence > 0
        };

        let rendered = self.render_node(node);
        if needs_parens && !matches!(node, AstNode::Const(_) | AstNode::Var(_)) {
            format!("({})", rendered)
        } else {
            rendered
        }
    }

    /// Render indentation
    fn render_indent(&self, buffer: &mut String) {
        for _ in 0..self.indent_level() {
            buffer.push('\t');
        }
    }

    /// Render a data type recursively, returning (base_type, array_dims)
    fn render_dtype_recursive(&self, dtype: &DType) -> (String, String) {
        match dtype {
            DType::Ptr(inner) => {
                let (base, dims) = self.render_dtype_recursive(inner);
                // void** のような多重ポインタを扱う
                if base.ends_with('*') || dims.is_empty() {
                    (format!("{}*", base), dims)
                } else {
                    (base, format!("(*{}){}", "", dims))
                }
            }
            DType::Vec(inner, size) => {
                let (base, dims) = self.render_dtype_recursive(inner);
                (base, format!("{}{}[{}]", dims, "", size))
            }
            _ => (self.render_scalar_dtype(dtype), "".to_string()),
        }
    }

    /// Render a variable declaration
    fn render_variable_decl(&self, decl: &VariableDecl) -> String {
        let mut buffer = String::new();
        if decl.constant {
            write!(buffer, "const ").unwrap();
        }
        let (base_type, array_dims) = self.render_dtype_recursive(&decl.dtype);
        write!(buffer, "{} {}{}", base_type, decl.name, array_dims).unwrap();
        buffer
    }

    /// Check if a node needs a semicolon when rendered as a statement
    fn needs_semicolon(node: &AstNode) -> bool {
        !matches!(
            node,
            AstNode::If { .. }
                | AstNode::Range { .. }
                | AstNode::Block { .. }
                | AstNode::Function { .. }
                | AstNode::Kernel { .. }
        )
    }

    /// Render a scope with declarations and statements
    fn render_scope(&mut self, scope: &Scope, statements: &[AstNode]) -> String {
        let mut buffer = String::new();
        let current_indent = self.indent_level();
        self.set_indent_level(current_indent + 1);
        writeln!(buffer, "{{").unwrap();

        // Track variables that need to be freed
        let mut malloc_vars = Vec::new();

        // Render declarations
        for decl in &scope.declarations {
            self.render_indent(&mut buffer);
            let decl_str = self.render_variable_decl(decl);
            writeln!(buffer, "{};", decl_str).unwrap();

            // For dynamic arrays (pointers with size_expr), emit allocation call
            if let Some(size_expr) = &decl.size_expr {
                if let DType::Ptr(inner) = &decl.dtype {
                    self.render_indent(&mut buffer);
                    let base_type = self.render_scalar_dtype(inner);
                    let size_code = self.render_node(size_expr);
                    let config = self.memory_config();

                    if config.needs_cast {
                        writeln!(
                            buffer,
                            "{} = ({}*){}(sizeof({}) * ({}));",
                            decl.name, base_type, config.alloc_fn, base_type, size_code
                        )
                        .unwrap();
                    } else {
                        writeln!(
                            buffer,
                            "{} = {}(sizeof({}) * ({}));",
                            decl.name, config.alloc_fn, base_type, size_code
                        )
                        .unwrap();
                    }
                    // Track this variable for cleanup
                    malloc_vars.push(decl.name.clone());
                }
            }
        }

        // Render statements
        for stmt in statements {
            self.render_indent(&mut buffer);
            let stmt_str = self.render_node(stmt);
            if Self::needs_semicolon(stmt) {
                writeln!(buffer, "{};", stmt_str).unwrap();
            } else {
                writeln!(buffer, "{}", stmt_str).unwrap();
            }
        }

        // Free dynamically allocated memory before closing scope
        if !malloc_vars.is_empty() {
            let config = self.memory_config();
            for var_name in malloc_vars {
                self.render_indent(&mut buffer);
                writeln!(buffer, "{}({});", config.dealloc_fn, var_name).unwrap();
            }
        }

        self.set_indent_level(current_indent);
        self.render_indent(&mut buffer);
        write!(buffer, "}}").unwrap();
        buffer
    }

    /// Render an AST node
    fn render_node(&mut self, node: &AstNode) -> String {
        let mut buffer = String::new();
        match node {
            AstNode::Const(c) => write!(buffer, "{}", self.render_const(c)).unwrap(),
            AstNode::Var(s) => write!(buffer, "{}", s).unwrap(),

            // Binary operators
            AstNode::Add(lhs, rhs) => {
                let prec = self.precedence(node);
                match &**rhs {
                    AstNode::Neg(negv) => {
                        // a + (-b) => a - b
                        write!(
                            buffer,
                            "{} - {}",
                            self.render_with_parens(lhs, prec, false),
                            self.render_with_parens(negv, prec, true)
                        )
                        .unwrap()
                    }
                    _ => write!(
                        buffer,
                        "{} + {}",
                        self.render_with_parens(lhs, prec, false),
                        self.render_with_parens(rhs, prec, true)
                    )
                    .unwrap(),
                }
            }
            AstNode::Mul(lhs, rhs) => {
                let prec = self.precedence(node);
                match &**rhs {
                    AstNode::Recip(recipv) => {
                        // a * recip(b) => a / b
                        write!(
                            buffer,
                            "{} / {}",
                            self.render_with_parens(lhs, prec, false),
                            self.render_with_parens(recipv, prec, true)
                        )
                        .unwrap()
                    }
                    _ => write!(
                        buffer,
                        "{} * {}",
                        self.render_with_parens(lhs, prec, false),
                        self.render_with_parens(rhs, prec, true)
                    )
                    .unwrap(),
                }
            }
            AstNode::Rem(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} % {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::BitAnd(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} & {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::BitOr(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} | {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::BitXor(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} ^ {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::Shl(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} << {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::Shr(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} >> {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::LessThan(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} < {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap();
            }
            AstNode::Eq(lhs, rhs) => {
                let prec = self.precedence(node);
                write!(
                    buffer,
                    "{} == {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap();
            }

            // Unary operators
            AstNode::Neg(v) => {
                let prec = self.precedence(node);
                write!(buffer, "-{}", self.render_with_parens(v, prec, false)).unwrap()
            }
            AstNode::BitNot(v) => {
                let prec = self.precedence(node);
                write!(buffer, "~{}", self.render_with_parens(v, prec, false)).unwrap()
            }
            AstNode::Recip(v) => write!(buffer, "(1 / {})", self.render_node(v)).unwrap(),

            // Math functions
            AstNode::Sin(v) => {
                let arg = self.render_node(v);
                write!(buffer, "{}", self.render_math_function("sin", vec![arg])).unwrap()
            }
            AstNode::Sqrt(v) => {
                let arg = self.render_node(v);
                write!(buffer, "{}", self.render_math_function("sqrt", vec![arg])).unwrap()
            }
            AstNode::Log2(v) => {
                let arg = self.render_node(v);
                write!(buffer, "{}", self.render_math_function("log2", vec![arg])).unwrap()
            }
            AstNode::Exp2(v) => {
                let arg = self.render_node(v);
                write!(buffer, "{}", self.render_math_function("exp2", vec![arg])).unwrap()
            }
            AstNode::Max(lhs, rhs) => {
                let arg1 = self.render_node(lhs);
                let arg2 = self.render_node(rhs);
                write!(
                    buffer,
                    "{}",
                    self.render_math_function("fmax", vec![arg1, arg2])
                )
                .unwrap()
            }

            // Assignment
            AstNode::Assign(var_name, rhs) => {
                write!(buffer, "{} = {}", var_name, self.render_node(rhs)).unwrap();
            }

            // Memory access
            AstNode::Load {
                target,
                index,
                vector_width,
            } => {
                let target_str = self.render_node(target);
                let index_str = self.render_node(index);
                match vector_width {
                    1 => {
                        // Scalar load: *(ptr + index)
                        write!(buffer, "*({} + {})", target_str, index_str).unwrap();
                    }
                    2 | 4 | 8 | 16 => {
                        // Vector load: *((float4*)(ptr + index))
                        write!(
                            buffer,
                            "*((float{}*)({} + {}))",
                            vector_width, target_str, index_str
                        )
                        .unwrap();
                    }
                    _ => panic!("Unsupported vector width: {}", vector_width),
                }
            }
            AstNode::Store {
                target,
                index,
                value,
                vector_width,
            } => {
                let target_str = self.render_node(target);
                let index_str = self.render_node(index);
                let value_str = self.render_node(value);
                match vector_width {
                    1 => {
                        // Scalar store: *(ptr + index) = value
                        write!(buffer, "*({} + {}) = {}", target_str, index_str, value_str)
                            .unwrap();
                    }
                    2 | 4 | 8 | 16 => {
                        // Vector store: *((float4*)(ptr + index)) = value
                        write!(
                            buffer,
                            "*((float{}*)({} + {})) = {}",
                            vector_width, target_str, index_str, value_str
                        )
                        .unwrap();
                    }
                    _ => panic!("Unsupported vector width: {}", vector_width),
                }
            }

            // Type cast
            AstNode::Cast { dtype, expr } => {
                let (base_type, dims) = self.render_dtype_recursive(dtype);
                let type_str = if !dims.is_empty() {
                    format!("{}*", base_type)
                } else {
                    base_type
                };
                write!(buffer, "({}){}", type_str, self.render_node(expr)).unwrap();
            }

            // Select (ternary operator)
            AstNode::Select {
                cond,
                true_val,
                false_val,
            } => {
                write!(
                    buffer,
                    "{} ? {} : {}",
                    self.render_node(cond),
                    self.render_node(true_val),
                    self.render_node(false_val)
                )
                .unwrap();
            }

            // Function call
            AstNode::CallFunction { name, args } => {
                let args_str = args
                    .iter()
                    .map(|arg| self.render_node(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(buffer, "{}({})", name, args_str).unwrap();
            }

            // Control flow
            AstNode::Range {
                counter_name,
                start,
                max,
                step,
                body,
                unroll,
            } => {
                let start_str = self.render_node(start);
                let max_str = self.render_node(max);
                let step_str = self.render_node(step);

                // Generate #pragma unroll if requested
                if let Some(factor) = unroll {
                    if *factor == 0 {
                        writeln!(buffer, "#pragma unroll").unwrap();
                    } else {
                        writeln!(buffer, "#pragma unroll {}", factor).unwrap();
                    }
                    self.render_indent(&mut buffer);
                }

                writeln!(
                    buffer,
                    "for (size_t {counter_name} = {start_str}; {counter_name} < {max_str}; {counter_name} += {step_str})"
                )
                .unwrap();

                match *body.as_ref() {
                    AstNode::Block { .. } => {
                        self.render_indent(&mut buffer);
                        write!(buffer, "{}", self.render_node(body.as_ref())).unwrap();
                    }
                    _ => {
                        let current_indent = self.indent_level();
                        self.set_indent_level(current_indent + 1);
                        self.render_indent(&mut buffer);
                        let body_str = self.render_node(body);
                        if Self::needs_semicolon(body) {
                            write!(buffer, "{};", body_str).unwrap();
                        } else {
                            write!(buffer, "{}", body_str).unwrap();
                        }
                        self.set_indent_level(current_indent);
                    }
                }
            }

            AstNode::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_str = self.render_node(condition);
                write!(buffer, "if ({})", cond_str).unwrap();

                // Render then branch
                match then_branch.as_ref() {
                    AstNode::Block { .. } => {
                        write!(buffer, " ").unwrap();
                        write!(buffer, "{}", self.render_node(then_branch)).unwrap();
                    }
                    _ => {
                        writeln!(buffer).unwrap();
                        let current_indent = self.indent_level();
                        self.set_indent_level(current_indent + 1);
                        self.render_indent(&mut buffer);
                        write!(buffer, "{};", self.render_node(then_branch)).unwrap();
                        self.set_indent_level(current_indent);
                    }
                }

                // Render else branch if present
                if let Some(else_br) = else_branch {
                    match then_branch.as_ref() {
                        AstNode::Block { .. } => {
                            write!(buffer, " else").unwrap();
                        }
                        _ => {
                            writeln!(buffer).unwrap();
                            self.render_indent(&mut buffer);
                            write!(buffer, "else").unwrap();
                        }
                    }

                    match else_br.as_ref() {
                        AstNode::Block { .. } => {
                            write!(buffer, " ").unwrap();
                            write!(buffer, "{}", self.render_node(else_br)).unwrap();
                        }
                        AstNode::If { .. } => {
                            // else if case
                            write!(buffer, " ").unwrap();
                            write!(buffer, "{}", self.render_node(else_br)).unwrap();
                        }
                        _ => {
                            writeln!(buffer).unwrap();
                            let current_indent = self.indent_level();
                            self.set_indent_level(current_indent + 1);
                            self.render_indent(&mut buffer);
                            write!(buffer, "{};", self.render_node(else_br)).unwrap();
                            self.set_indent_level(current_indent);
                        }
                    }
                }
            }

            AstNode::Block { scope, statements } => {
                write!(buffer, "{}", self.render_scope(scope, statements)).unwrap();
            }

            AstNode::Barrier => {
                write!(buffer, "{}", self.render_barrier()).unwrap();
            }

            AstNode::Function {
                name,
                scope,
                statements,
                arguments,
                return_type,
            } => {
                let (ret_type, _) = self.render_dtype_recursive(return_type);
                let args_str = arguments
                    .iter()
                    .map(|(arg_name, dtype)| {
                        let (base_type, array_dims) = self.render_dtype_recursive(dtype);
                        format!("{} {}{}", base_type, arg_name, array_dims)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(buffer, "{} {}({})", ret_type, name, args_str).unwrap();
                write!(buffer, "{}", self.render_scope(scope, statements)).unwrap();
            }

            AstNode::Kernel {
                name,
                scope,
                statements,
                arguments,
                return_type,
                ..
            } => {
                // Render kernel as a regular function
                // GPU execution model (thread IDs, barriers) is emulated in CPU
                let (ret_type, _) = self.render_dtype_recursive(return_type);
                let args_str = arguments
                    .iter()
                    .map(|(arg_name, dtype)| {
                        let (base_type, array_dims) = self.render_dtype_recursive(dtype);
                        format!("{} {}{}", base_type, arg_name, array_dims)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(buffer, "{} {}({})", ret_type, name, args_str).unwrap();

                // Render kernel scope and statements
                let current_indent = self.indent_level();
                self.set_indent_level(current_indent + 1);
                writeln!(buffer, "{{").unwrap();

                // Render thread ID declarations (from KernelScope)
                for thread_id_decl in &scope.thread_ids {
                    self.render_indent(&mut buffer);
                    writeln!(buffer, "size_t {}[3];", thread_id_decl.name).unwrap();
                }

                // Render regular variable declarations
                for decl in &scope.declarations {
                    self.render_indent(&mut buffer);
                    let decl_str = self.render_variable_decl(decl);
                    writeln!(buffer, "{};", decl_str).unwrap();

                    // Handle dynamic allocations
                    if let Some(size_expr) = &decl.size_expr {
                        if let DType::Ptr(inner) = &decl.dtype {
                            self.render_indent(&mut buffer);
                            let base_type = self.render_scalar_dtype(inner);
                            let size_code = self.render_node(size_expr);
                            let config = self.memory_config();

                            if config.needs_cast {
                                writeln!(
                                    buffer,
                                    "{} = ({}*){}(sizeof({}) * ({}));",
                                    decl.name, base_type, config.alloc_fn, base_type, size_code
                                )
                                .unwrap();
                            } else {
                                writeln!(
                                    buffer,
                                    "{} = {}(sizeof({}) * ({}));",
                                    decl.name, config.alloc_fn, base_type, size_code
                                )
                                .unwrap();
                            }
                        }
                    }
                }

                // Render statements
                for stmt in statements {
                    self.render_indent(&mut buffer);
                    let stmt_str = self.render_node(stmt);
                    if Self::needs_semicolon(stmt) {
                        writeln!(buffer, "{};", stmt_str).unwrap();
                    } else {
                        writeln!(buffer, "{}", stmt_str).unwrap();
                    }
                }

                // Free dynamic allocations
                for decl in &scope.declarations {
                    if decl.size_expr.is_some() {
                        if let DType::Ptr(_) = &decl.dtype {
                            self.render_indent(&mut buffer);
                            let config = self.memory_config();
                            writeln!(buffer, "{}({});", config.dealloc_fn, decl.name).unwrap();
                        }
                    }
                }

                self.set_indent_level(current_indent);
                self.render_indent(&mut buffer);
                write!(buffer, "}}").unwrap();
            }

            AstNode::CallKernel {
                name,
                args,
                global_size,
                ..
            } => {
                // Render kernel call as OpenMP parallel for loop
                // global_size[0] is the primary dimension to parallelize
                let total_threads_str = self.render_node(&global_size[0]);

                // Generate unique loop counter name to avoid conflicts
                let loop_counter = format!(
                    "__kernel_idx_{}",
                    name.replace(|c: char| !c.is_alphanumeric(), "_")
                );

                writeln!(buffer, "#pragma omp parallel for").unwrap();
                self.render_indent(&mut buffer);
                writeln!(
                    buffer,
                    "for (size_t {} = 0; {} < {}; {}++)",
                    loop_counter, loop_counter, total_threads_str, loop_counter
                )
                .unwrap();
                self.render_indent(&mut buffer);
                writeln!(buffer, "{{").unwrap();

                let current_indent = self.indent_level();
                self.set_indent_level(current_indent + 1);

                // Render kernel function call with thread index
                self.render_indent(&mut buffer);
                let args_str = args
                    .iter()
                    .map(|arg| self.render_node(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(buffer, "{}({});", name, args_str).unwrap();

                self.set_indent_level(current_indent);
                self.render_indent(&mut buffer);
                write!(buffer, "}}").unwrap();
            }

            AstNode::Program { .. } | AstNode::Drop(_) | AstNode::Rand | AstNode::Capture(_) => {
                // These nodes should be handled at a higher level or have no rendering
                panic!("Unexpected node type in render_node: {:?}", node)
            }
        }
        buffer
    }
}
