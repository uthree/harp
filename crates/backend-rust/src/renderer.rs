//! Rust language renderer for Eclat
//!
//! This module provides a renderer that generates unsafe Rust code
//! that can be compiled as a cdylib for dynamic loading.
//!
//! The generated code uses `extern "C"` ABI for compatibility with libloading.

use eclat::ast::{AstNode, DType, Literal, ParallelInfo, ParallelKind, Scope, VarDecl};
use eclat::backend::Renderer;
use eclat::backend::renderer::CLikeRenderer;

/// Rust source code representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RustCode(String);

impl RustCode {
    /// Create new RustCode
    pub fn new(code: String) -> Self {
        Self(code)
    }

    /// Get reference to inner String
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Get inner String (consumes self)
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Get code length in bytes
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if code is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if code contains a pattern
    pub fn contains(&self, pat: &str) -> bool {
        self.0.contains(pat)
    }
}

impl From<String> for RustCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<RustCode> for String {
    fn from(code: RustCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for RustCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for RustCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Rust code renderer
///
/// Generates unsafe Rust code with extern "C" ABI for dynamic loading.
#[derive(Debug, Clone, Default)]
pub struct RustRenderer {
    indent_level: usize,
    /// Flag indicating whether rayon is needed in the generated code
    needs_rayon: bool,
}

impl RustRenderer {
    /// Create a new RustRenderer
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            needs_rayon: false,
        }
    }

    /// Render a Rust literal
    fn render_literal_rust(&self, lit: &Literal) -> String {
        match lit {
            Literal::Bool(v) => {
                if *v {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            Literal::I8(v) => format!("{}i8", v),
            Literal::I16(v) => format!("{}i16", v),
            Literal::I32(v) => format!("{}i32", v),
            Literal::I64(v) => format!("{}i64", v),
            Literal::U8(v) => format!("{}u8", v),
            Literal::U16(v) => format!("{}u16", v),
            Literal::U32(v) => format!("{}u32", v),
            Literal::U64(v) => format!("{}u64", v),
            Literal::F32(v) => {
                if v.is_nan() {
                    "f32::NAN".to_string()
                } else if v.is_infinite() {
                    if v.is_sign_positive() {
                        "f32::INFINITY".to_string()
                    } else {
                        "f32::NEG_INFINITY".to_string()
                    }
                } else {
                    let s = format!("{}", v);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        format!("{}.0f32", s)
                    } else {
                        format!("{}f32", s)
                    }
                }
            }
            Literal::F64(v) => {
                if v.is_nan() {
                    "f64::NAN".to_string()
                } else if v.is_infinite() {
                    if v.is_sign_positive() {
                        "f64::INFINITY".to_string()
                    } else {
                        "f64::NEG_INFINITY".to_string()
                    }
                } else {
                    let s = format!("{}", v);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        format!("{}.0f64", s)
                    } else {
                        format!("{}f64", s)
                    }
                }
            }
            Literal::Complex32(re, im) => {
                format!("/* complex32({}, {}) */", re, im)
            }
            Literal::Complex64(re, im) => {
                format!("/* complex64({}, {}) */", re, im)
            }
        }
    }

    /// Render expression in Rust syntax
    fn render_expr_rust(&self, node: &AstNode) -> String {
        match node {
            AstNode::Wildcard(name) => name.clone(),
            AstNode::Const(lit) => self.render_literal_rust(lit),
            AstNode::Var(name) => name.clone(),
            AstNode::Add(left, right) => {
                format!(
                    "({} + {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::Mul(left, right) => {
                format!(
                    "({} * {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::Max(left, right) => {
                format!(
                    "({}).max({})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::Rem(left, right) => {
                format!(
                    "({} % {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::Idiv(left, right) => {
                format!(
                    "({} / {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::Recip(operand) => {
                format!("(1.0f32 / {})", self.render_expr_rust(operand))
            }
            AstNode::Sqrt(operand) => {
                format!("({}).sqrt()", self.render_expr_rust(operand))
            }
            AstNode::Log2(operand) => {
                format!("({}).log2()", self.render_expr_rust(operand))
            }
            AstNode::Exp2(operand) => {
                format!("({}).exp2()", self.render_expr_rust(operand))
            }
            AstNode::Sin(operand) => {
                format!("({}).sin()", self.render_expr_rust(operand))
            }
            AstNode::Floor(operand) => {
                format!("({}).floor()", self.render_expr_rust(operand))
            }
            AstNode::Rand => "/* rand not supported */0.0f32".to_string(),
            AstNode::BitwiseAnd(left, right) => {
                format!(
                    "({} & {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::BitwiseOr(left, right) => {
                format!(
                    "({} | {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::BitwiseXor(left, right) => {
                format!(
                    "({} ^ {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::BitwiseNot(operand) => {
                format!("(!{})", self.render_expr_rust(operand))
            }
            AstNode::LeftShift(left, right) => {
                format!(
                    "({} << {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::RightShift(left, right) => {
                format!(
                    "({} >> {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::Lt(left, right) => {
                format!(
                    "({} < {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::And(left, right) => {
                format!(
                    "({} && {})",
                    self.render_expr_rust(left),
                    self.render_expr_rust(right)
                )
            }
            AstNode::Not(operand) => {
                format!("(!{})", self.render_expr_rust(operand))
            }
            AstNode::Select {
                cond,
                then_val,
                else_val,
            } => {
                format!(
                    "(if {} {{ {} }} else {{ {} }})",
                    self.render_expr_rust(cond),
                    self.render_expr_rust(then_val),
                    self.render_expr_rust(else_val)
                )
            }
            AstNode::Cast(operand, dtype) => {
                format!(
                    "({} as {})",
                    self.render_expr_rust(operand),
                    self.render_dtype_backend(dtype)
                )
            }
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype: _,
            } => {
                if *count == 1 {
                    // Scalar load: *ptr.add(offset as usize)
                    format!(
                        "*{}.add({} as usize)",
                        self.render_expr_rust(ptr),
                        self.render_expr_rust(offset)
                    )
                } else {
                    // Vector load (not fully supported)
                    format!(
                        "/* vec{} load */ *{}.add({} as usize)",
                        count,
                        self.render_expr_rust(ptr),
                        self.render_expr_rust(offset)
                    )
                }
            }
            AstNode::Call { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.render_expr_rust(a)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            AstNode::Return { value } => {
                format!("return {}", self.render_expr_rust(value))
            }
            AstNode::Fma { a, b, c } => {
                format!(
                    "({}).mul_add({}, {})",
                    self.render_expr_rust(a),
                    self.render_expr_rust(b),
                    self.render_expr_rust(c)
                )
            }
            AstNode::AtomicAdd {
                ptr,
                offset,
                value,
                dtype: _,
            } => {
                // Sequential execution - just regular addition
                format!(
                    "{{ let p = {}.add({} as usize); *p += {}; *p }}",
                    self.render_expr_rust(ptr),
                    self.render_expr_rust(offset),
                    self.render_expr_rust(value)
                )
            }
            AstNode::AtomicMax {
                ptr,
                offset,
                value,
                dtype: _,
            } => {
                format!(
                    "{{ let p = {}.add({} as usize); *p = (*p).max({}); *p }}",
                    self.render_expr_rust(ptr),
                    self.render_expr_rust(offset),
                    self.render_expr_rust(value)
                )
            }
            AstNode::Real(operand) => {
                format!("/* real */ {}", self.render_expr_rust(operand))
            }
            AstNode::Imag(_operand) => "/* imag */ 0.0f32".to_string(),
            AstNode::Conj(operand) => {
                format!("/* conj */ {}", self.render_expr_rust(operand))
            }
            AstNode::MakeComplex { re, im: _ } => {
                format!("/* make_complex */ {}", self.render_expr_rust(re))
            }
            _ => format!("/* unsupported: {:?} */", std::mem::discriminant(node)),
        }
    }

    /// Render statement in Rust syntax
    fn render_statement_rust(&mut self, node: &AstNode) -> String {
        match node {
            AstNode::Store { ptr, offset, value } => {
                format!(
                    "{}*{}.add({} as usize) = {};",
                    self.indent(),
                    self.render_expr_rust(ptr),
                    self.render_expr_rust(offset),
                    self.render_expr_rust(value)
                )
            }
            AstNode::Assign { var, value } => {
                format!(
                    "{}{} = {};",
                    self.indent(),
                    var,
                    self.render_expr_rust(value)
                )
            }
            AstNode::Block { statements, scope } => self.render_block_rust(statements, scope),
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
                parallel,
            } => self.render_range_rust(var, start, step, stop, body, parallel),
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => self.render_if_rust(condition, then_body, else_body.as_deref()),
            AstNode::Return { value } => {
                format!("{}return {};", self.indent(), self.render_expr_rust(value))
            }
            AstNode::Barrier => {
                format!("{}// barrier (no-op in Rust)", self.indent())
            }
            AstNode::Call { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.render_expr_rust(a)).collect();
                format!("{}{}({});", self.indent(), name, arg_strs.join(", "))
            }
            _ => {
                // Expression statement
                format!("{}{};", self.indent(), self.render_expr_rust(node))
            }
        }
    }

    /// Render block in Rust syntax
    fn render_block_rust(&mut self, statements: &[AstNode], scope: &Scope) -> String {
        let mut result = String::new();

        // Variable declarations at block start
        for var_decl in scope.local_variables() {
            let type_str = self.render_dtype_backend(&var_decl.dtype);
            // Use Default trait for initialization
            let default_val = match &var_decl.dtype {
                DType::F32 => "0.0f32",
                DType::F64 => "0.0f64",
                DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::Int => "0",
                DType::U8 | DType::U16 | DType::U32 | DType::U64 => "0",
                DType::Bool => "false",
                _ => "Default::default()",
            };
            result.push_str(&format!(
                "{}let mut {}: {} = {};\n",
                self.indent(),
                var_decl.name,
                type_str,
                default_val
            ));
        }

        // Render statements
        for stmt in statements {
            result.push_str(&self.render_statement_rust(stmt));
            result.push('\n');
        }
        result
    }

    /// Render range loop in Rust syntax
    fn render_range_rust(
        &mut self,
        var: &str,
        start: &AstNode,
        step: &AstNode,
        stop: &AstNode,
        body: &AstNode,
        parallel: &ParallelInfo,
    ) -> String {
        let mut result = String::new();

        // Check if step is 1
        let step_is_one = matches!(
            step,
            AstNode::Const(Literal::I64(1))
                | AstNode::Const(Literal::I32(1))
                | AstNode::Const(Literal::U64(1))
                | AstNode::Const(Literal::U32(1))
        );

        let start_str = self.render_expr_rust(start);
        let stop_str = self.render_expr_rust(stop);
        let step_str = self.render_expr_rust(step);

        // Check if this is a parallel Rayon loop
        let use_rayon = parallel.is_parallel
            && (parallel.kind == ParallelKind::Rayon || parallel.kind == ParallelKind::OpenMP);

        if use_rayon {
            // Use Rayon parallel iterator
            self.needs_rayon = true;

            if step_is_one {
                // (start..stop).into_par_iter().for_each(|var| { ... });
                result.push_str(&format!(
                    "{}({}..{}).into_par_iter().for_each(|{}| {{\n",
                    self.indent(),
                    start_str,
                    stop_str,
                    var
                ));
            } else {
                // (start..stop).into_par_iter().step_by(step).for_each(|var| { ... });
                result.push_str(&format!(
                    "{}({}..{}).into_par_iter().step_by({} as usize).for_each(|{}| {{\n",
                    self.indent(),
                    start_str,
                    stop_str,
                    step_str,
                    var
                ));
            }

            self.inc_indent();
            let body_str = self.render_statement_rust(body);
            result.push_str(&body_str);
            if !body_str.ends_with('\n') {
                result.push('\n');
            }
            self.dec_indent();
            result.push_str(&format!("{}}});", self.indent()));
        } else {
            // Sequential loop
            if step_is_one {
                // for var in start..stop { }
                result.push_str(&format!(
                    "{}for {} in {}..{} {{\n",
                    self.indent(),
                    var,
                    start_str,
                    stop_str
                ));
            } else {
                // for var in (start..stop).step_by(step as usize) { }
                result.push_str(&format!(
                    "{}for {} in ({}..{}).step_by({} as usize) {{\n",
                    self.indent(),
                    var,
                    start_str,
                    stop_str,
                    step_str
                ));
            }

            self.inc_indent();
            let body_str = self.render_statement_rust(body);
            result.push_str(&body_str);
            if !body_str.ends_with('\n') {
                result.push('\n');
            }
            self.dec_indent();
            result.push_str(&format!("{}}}", self.indent()));
        }
        result
    }

    /// Render if statement in Rust syntax
    fn render_if_rust(
        &mut self,
        condition: &AstNode,
        then_body: &AstNode,
        else_body: Option<&AstNode>,
    ) -> String {
        let mut result = String::new();
        result.push_str(&format!(
            "{}if {} {{\n",
            self.indent(),
            self.render_expr_rust(condition)
        ));

        self.inc_indent();
        let then_str = self.render_statement_rust(then_body);
        result.push_str(&then_str);
        if !then_str.ends_with('\n') {
            result.push('\n');
        }
        self.dec_indent();
        result.push_str(&format!("{}}}", self.indent()));

        if let Some(else_b) = else_body {
            result.push_str(" else {\n");
            self.inc_indent();
            let else_str = self.render_statement_rust(else_b);
            result.push_str(&else_str);
            if !else_str.ends_with('\n') {
                result.push('\n');
            }
            self.dec_indent();
            result.push_str(&format!("{}}}", self.indent()));
        }

        result
    }

    /// Render function parameter in Rust syntax
    fn render_param_rust(&self, param: &VarDecl) -> String {
        let type_str = self.render_dtype_backend(&param.dtype);
        format!("{}: {}", param.name, type_str)
    }

    /// Render the entire program in Rust syntax
    fn render_program_rust(&mut self, program: &AstNode) -> String {
        let AstNode::Program { functions, .. } = program else {
            panic!("Expected AstNode::Program");
        };

        // First render functions (this may set needs_rayon)
        let mut functions_code = String::new();
        for func in functions {
            functions_code.push_str(&self.render_function_node_rust(func));
            functions_code.push('\n');
        }

        // Now build the final result with header (can now check needs_rayon)
        let mut result = String::new();
        result.push_str(&self.render_header());
        result.push_str(&functions_code);

        result
    }

    /// Render function/kernel node in Rust syntax
    fn render_function_node_rust(&mut self, func_node: &AstNode) -> String {
        let (name, params, return_type, body, _is_kernel) = match func_node {
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => (name, params, return_type, body, false),
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                ..
            } => (name, params, return_type, body, true),
            _ => panic!("Expected AstNode::Function or AstNode::Kernel"),
        };

        let func_name = name.as_ref().map(|s| s.as_str()).unwrap_or("anonymous");

        let mut result = String::new();

        // Function attributes
        result.push_str("#[no_mangle]\n");
        result.push_str("pub unsafe extern \"C\" fn ");
        result.push_str(func_name);
        result.push('(');

        // Parameters
        let rendered_params: Vec<String> =
            params.iter().map(|p| self.render_param_rust(p)).collect();
        result.push_str(&rendered_params.join(", "));

        result.push(')');

        // Return type
        let ret_type = self.render_dtype_backend(return_type);
        if ret_type != "()" {
            result.push_str(" -> ");
            result.push_str(&ret_type);
        }

        result.push_str(" {\n");

        // Function body
        self.inc_indent();
        let body_str = self.render_statement_rust(body);
        result.push_str(&body_str);
        if !body_str.ends_with('\n') {
            result.push('\n');
        }
        self.dec_indent();
        result.push_str("}\n");
        result
    }
}

impl Renderer for RustRenderer {
    type CodeRepr = RustCode;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut r = self.clone();
        RustCode::new(r.render_program_rust(program))
    }

    fn is_available(&self) -> bool {
        true
    }
}

// CLikeRenderer implementation (required for Pipeline compatibility)
impl CLikeRenderer for RustRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        match dtype {
            DType::Void => "()".to_string(),
            DType::Bool => "bool".to_string(),
            DType::I8 => "i8".to_string(),
            DType::I16 => "i16".to_string(),
            DType::I32 => "i32".to_string(),
            DType::I64 => "i64".to_string(),
            DType::U8 => "u8".to_string(),
            DType::U16 => "u16".to_string(),
            DType::U32 => "u32".to_string(),
            DType::U64 => "u64".to_string(),
            DType::F32 => "f32".to_string(),
            DType::F64 => "f64".to_string(),
            DType::Int => "i64".to_string(),
            DType::Ptr(inner) => format!("*mut {}", self.render_dtype_backend(inner)),
            DType::Vec(inner, size) => {
                format!("[{}; {}]", self.render_dtype_backend(inner), size)
            }
            DType::Tuple(types) => {
                if types.is_empty() {
                    "()".to_string()
                } else {
                    let inner: Vec<String> =
                        types.iter().map(|t| self.render_dtype_backend(t)).collect();
                    format!("({})", inner.join(", "))
                }
            }
            DType::Unknown => "/* unknown */".to_string(),
            DType::Complex32 => "/* Complex32 */ (f32, f32)".to_string(),
            DType::Complex64 => "/* Complex64 */ (f64, f64)".to_string(),
        }
    }

    fn render_barrier_backend(&self) -> String {
        "// barrier (no-op in Rust)".to_string()
    }

    fn render_header(&self) -> String {
        let mut header = String::from(
            r#"// Generated Rust code by Eclat
// This code is compiled as cdylib for dynamic loading

#![allow(unused_unsafe)]
#![allow(unused_parens)]
#![allow(unused_variables)]
#![allow(dead_code)]

"#,
        );

        // Add rayon import if parallel loops are used
        if self.needs_rayon {
            header.push_str("use rayon::prelude::*;\n\n");
        }

        header
    }

    fn render_function_qualifier(&self, _is_kernel: bool) -> String {
        "#[no_mangle]\npub unsafe extern \"C\"".to_string()
    }

    fn render_param_attribute(&self, param: &VarDecl, _is_kernel: bool) -> String {
        let type_str = self.render_dtype_backend(&param.dtype);
        format!("{}: {}", param.name, type_str)
    }

    fn render_thread_var_declarations(&self, _params: &[VarDecl], _indent: &str) -> String {
        // No thread-local variables in sequential Rust
        String::new()
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        // Use Rust method syntax
        match name {
            "max" => format!("({}).max({})", args[0], args[1]),
            "min" => format!("({}).min({})", args[0], args[1]),
            "sqrt" => format!("({}).sqrt()", args[0]),
            "log2" => format!("({}).log2()", args[0]),
            "exp2" => format!("({}).exp2()", args[0]),
            "sin" => format!("({}).sin()", args[0]),
            "cos" => format!("({}).cos()", args[0]),
            "floor" => format!("({}).floor()", args[0]),
            "ceil" => format!("({}).ceil()", args[0]),
            "abs" => format!("({}).abs()", args[0]),
            "exp" => format!("({}).exp()", args[0]),
            "log" => format!("({}).ln()", args[0]),
            "pow" => format!("({}).powf({})", args[0], args[1]),
            "tan" => format!("({}).tan()", args[0]),
            "tanh" => format!("({}).tanh()", args[0]),
            _ => format!("{}({})", name, args.join(", ")),
        }
    }

    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, _dtype: &DType) -> String {
        // Sequential execution - just regular addition
        format!(
            "{{ let p = {}.add({} as usize); *p += {}; *p }}",
            ptr, offset, value
        )
    }

    fn render_atomic_max(&self, ptr: &str, offset: &str, value: &str, _dtype: &DType) -> String {
        format!(
            "{{ let p = {}.add({} as usize); *p = (*p).max({}); *p }}",
            ptr, offset, value
        )
    }
}

/// Implementation of KernelSourceRenderer for Rust backend
impl eclat::backend::pipeline::KernelSourceRenderer for RustRenderer {
    fn render_kernel_source(&mut self, program: &AstNode) -> String {
        self.render_program_rust(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eclat::ast::Literal;

    #[test]
    fn test_render_header() {
        let renderer = RustRenderer::new();
        let header = renderer.render_header();
        assert!(header.contains("#![allow(unused_unsafe)]"));
        assert!(header.contains("cdylib"));
    }

    #[test]
    fn test_render_dtype() {
        let renderer = RustRenderer::new();
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "f32");
        assert_eq!(renderer.render_dtype_backend(&DType::F64), "f64");
        assert_eq!(renderer.render_dtype_backend(&DType::I32), "i32");
        assert_eq!(renderer.render_dtype_backend(&DType::I64), "i64");
        assert_eq!(
            renderer.render_dtype_backend(&DType::Ptr(Box::new(DType::F32))),
            "*mut f32"
        );
    }

    #[test]
    fn test_render_math_functions() {
        let renderer = RustRenderer::new();
        assert_eq!(
            renderer.render_math_func("sqrt", &["x".to_string()]),
            "(x).sqrt()"
        );
        assert_eq!(
            renderer.render_math_func("max", &["a".to_string(), "b".to_string()]),
            "(a).max(b)"
        );
    }

    #[test]
    fn test_render_binary_ops() {
        let renderer = RustRenderer::new();
        let a = AstNode::Const(Literal::F32(1.0));
        let b = AstNode::Const(Literal::F32(2.0));

        let add = AstNode::Add(Box::new(a.clone()), Box::new(b.clone()));
        assert_eq!(renderer.render_expr_rust(&add), "(1.0f32 + 2.0f32)");

        let mul = AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()));
        assert_eq!(renderer.render_expr_rust(&mul), "(1.0f32 * 2.0f32)");
    }

    #[test]
    fn test_render_literals() {
        let renderer = RustRenderer::new();
        assert_eq!(renderer.render_literal_rust(&Literal::F32(1.5)), "1.5f32");
        assert_eq!(renderer.render_literal_rust(&Literal::I64(42)), "42i64");
        assert_eq!(renderer.render_literal_rust(&Literal::Bool(true)), "true");
    }

    #[test]
    fn test_rustcode_conversions() {
        let code = RustCode::new("fn main() {}".to_string());
        assert_eq!(code.as_str(), "fn main() {}");
        assert_eq!(code.len(), 12);
        assert!(!code.is_empty());
        assert!(code.contains("main"));

        let string: String = code.into();
        assert_eq!(string, "fn main() {}");
    }

    #[test]
    fn test_render_parallel_range() {
        let mut renderer = RustRenderer::new();

        // Sequential range
        let seq_parallel = ParallelInfo {
            is_parallel: false,
            kind: ParallelKind::Sequential,
            reductions: vec![],
        };
        let seq_code = renderer.render_range_rust(
            "i",
            &AstNode::Const(Literal::I64(0)),
            &AstNode::Const(Literal::I64(1)),
            &AstNode::Const(Literal::I64(100)),
            &AstNode::Block {
                statements: vec![],
                scope: Box::new(Scope::default()),
            },
            &seq_parallel,
        );
        assert!(seq_code.contains("for i in 0i64..100i64"));
        assert!(!seq_code.contains("into_par_iter"));

        // Parallel range with Rayon
        let par_parallel = ParallelInfo {
            is_parallel: true,
            kind: ParallelKind::Rayon,
            reductions: vec![],
        };
        let par_code = renderer.render_range_rust(
            "i",
            &AstNode::Const(Literal::I64(0)),
            &AstNode::Const(Literal::I64(1)),
            &AstNode::Const(Literal::I64(100)),
            &AstNode::Block {
                statements: vec![],
                scope: Box::new(Scope::default()),
            },
            &par_parallel,
        );
        assert!(par_code.contains("into_par_iter"));
        assert!(par_code.contains("for_each"));
        assert!(renderer.needs_rayon);
    }

    #[test]
    fn test_render_header_with_rayon() {
        let mut renderer = RustRenderer::new();

        // Without rayon
        let header1 = renderer.render_header();
        assert!(!header1.contains("use rayon"));

        // With rayon
        renderer.needs_rayon = true;
        let header2 = renderer.render_header();
        assert!(header2.contains("use rayon::prelude::*"));
    }
}
