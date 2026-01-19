//! OpenMP renderer for Eclat
//!
//! This module provides a renderer that generates C code with OpenMP pragmas
//! for parallel execution on multi-core CPUs.

use eclat::ast::{AstNode, DType, ParallelInfo, ParallelKind, ReductionOp, VarDecl};
use eclat::backend::renderer::CLikeRenderer;
use eclat::backend::Renderer;

/// OpenMP C source code representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenMPCode(String);

impl OpenMPCode {
    /// Create new OpenMPCode
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

impl From<String> for OpenMPCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<OpenMPCode> for String {
    fn from(code: OpenMPCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for OpenMPCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for OpenMPCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// OpenMP code renderer
///
/// Generates C code with OpenMP pragmas for parallel execution.
#[derive(Debug, Clone, Default)]
pub struct OpenMPRenderer {
    indent_level: usize,
}

impl OpenMPRenderer {
    /// Create a new OpenMPRenderer
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// Render reduction operator to OpenMP syntax
    fn render_reduction_op(&self, op: &ReductionOp) -> &'static str {
        match op {
            ReductionOp::Add => "+",
            ReductionOp::Mul => "*",
            ReductionOp::Max => "max",
            ReductionOp::Min => "min",
        }
    }
}

impl Renderer for OpenMPRenderer {
    type CodeRepr = OpenMPCode;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut r = self.clone();
        OpenMPCode::new(r.render_program_clike(program))
    }

    fn is_available(&self) -> bool {
        // Check if OpenMP is available by testing compiler
        std::process::Command::new("cc")
            .args(["-fopenmp", "--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

impl CLikeRenderer for OpenMPRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        match dtype {
            DType::Void => "void".to_string(),
            DType::Bool => "unsigned char".to_string(),
            DType::I8 => "signed char".to_string(),
            DType::I16 => "short".to_string(),
            DType::I32 => "int".to_string(),
            DType::I64 => "long long".to_string(),
            DType::U8 => "unsigned char".to_string(),
            DType::U16 => "unsigned short".to_string(),
            DType::U32 => "unsigned int".to_string(),
            DType::U64 => "unsigned long long".to_string(),
            DType::F16 => "_Float16".to_string(),
            DType::BF16 => "__bf16".to_string(),
            DType::F32 => "float".to_string(),
            DType::F64 => "double".to_string(),
            DType::Int => "long long".to_string(),
            DType::Ptr(inner) => format!("{}*", self.render_dtype_backend(inner)),
            DType::Vec(inner, size) => format!("{}[{}]", self.render_dtype_backend(inner), size),
            DType::Tuple(types) => {
                if types.is_empty() {
                    "void".to_string()
                } else {
                    format!("/* tuple_{} */", types.len())
                }
            }
            DType::Unknown => "/* unknown */".to_string(),
            DType::Complex32 => "float _Complex".to_string(),
            DType::Complex64 => "double _Complex".to_string(),
        }
    }

    fn render_barrier_backend(&self) -> String {
        // OpenMP barrier
        "#pragma omp barrier".to_string()
    }

    fn render_header(&self) -> String {
        r#"/* Generated C code with OpenMP by Eclat */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

"#
        .to_string()
    }

    fn render_function_qualifier(&self, _is_kernel: bool) -> String {
        String::new()
    }

    fn render_param_attribute(&self, param: &VarDecl, _is_kernel: bool) -> String {
        let type_str = self.render_dtype_backend(&param.dtype);
        format!("{} {}", type_str, param.name)
    }

    fn render_thread_var_declarations(&self, _params: &[VarDecl], _indent: &str) -> String {
        String::new()
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        match name {
            "max" => format!("fmaxf({}, {})", args[0], args[1]),
            "min" => format!("fminf({}, {})", args[0], args[1]),
            "sqrt" => format!("sqrtf({})", args[0]),
            "log2" => format!("log2f({})", args[0]),
            "exp2" => format!("exp2f({})", args[0]),
            "sin" => format!("sinf({})", args[0]),
            "cos" => format!("cosf({})", args[0]),
            "floor" => format!("floorf({})", args[0]),
            "ceil" => format!("ceilf({})", args[0]),
            "abs" => format!("fabsf({})", args[0]),
            "exp" => format!("expf({})", args[0]),
            "log" => format!("logf({})", args[0]),
            "pow" => format!("powf({}, {})", args[0], args[1]),
            "tan" => format!("tanf({})", args[0]),
            "tanh" => format!("tanhf({})", args[0]),
            _ => format!("{}({})", name, args.join(", ")),
        }
    }

    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, _dtype: &DType) -> String {
        // Use OpenMP atomic directive
        format!("#pragma omp atomic\n{}[{}] += {}", ptr, offset, value)
    }

    fn render_atomic_max(&self, ptr: &str, offset: &str, value: &str, _dtype: &DType) -> String {
        // OpenMP 3.1+ supports atomic capture for max
        format!(
            "{{ float __tmp = {}; \n#pragma omp critical\n{{ if ({} > {}[{}]) {}[{}] = {}; }} }}",
            value, "__tmp", ptr, offset, ptr, offset, "__tmp"
        )
    }

    fn render_range(
        &mut self,
        var: &str,
        start: &AstNode,
        step: &AstNode,
        stop: &AstNode,
        body: &AstNode,
        parallel: &ParallelInfo,
    ) -> String {
        let indent = self.indent();
        let start_str = self.render_expr(start);
        let stop_str = self.render_expr(stop);
        let step_str = self.render_expr(step);

        let mut result = String::new();

        // Add OpenMP pragma if parallel
        if parallel.is_parallel && parallel.kind == ParallelKind::OpenMP {
            result.push_str(&format!("{}#pragma omp parallel for", indent));

            // Add reduction clauses
            for (var_name, op) in &parallel.reductions {
                let op_str = self.render_reduction_op(op);
                result.push_str(&format!(" reduction({}:{})", op_str, var_name));
            }

            result.push('\n');
        }

        // Render the for loop
        result.push_str(&format!(
            "{}for (long long {} = {}; {} < {}; {} += {}) {{\n",
            indent, var, start_str, var, stop_str, var, step_str
        ));

        *self.indent_level_mut() += 1;
        result.push_str(&self.render_statement(body));
        *self.indent_level_mut() -= 1;

        result.push_str(&format!("{}}}\n", indent));

        result
    }
}

/// Implementation of KernelSourceRenderer for OpenMP backend
impl eclat::backend::pipeline::KernelSourceRenderer for OpenMPRenderer {
    fn render_kernel_source(&mut self, program: &AstNode) -> String {
        self.render_program_clike(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use eclat::ast::Literal;

    #[test]
    fn test_render_header() {
        let renderer = OpenMPRenderer::new();
        let header = renderer.render_header();
        assert!(header.contains("#include <omp.h>"));
        assert!(header.contains("OpenMP"));
    }

    #[test]
    fn test_render_dtype() {
        let renderer = OpenMPRenderer::new();
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::F64), "double");
        assert_eq!(renderer.render_dtype_backend(&DType::I32), "int");
    }

    #[test]
    fn test_render_parallel_for() {
        let mut renderer = OpenMPRenderer::new();
        let body = AstNode::Const(Literal::I64(0));
        let parallel = ParallelInfo {
            is_parallel: true,
            kind: ParallelKind::OpenMP,
            reductions: vec![],
        };

        let code = renderer.render_range(
            "i",
            &AstNode::Const(Literal::I64(0)),
            &AstNode::Var("n".to_string()),
            &AstNode::Const(Literal::I64(1)),
            &body,
            &parallel,
        );

        assert!(code.contains("#pragma omp parallel for"));
        assert!(code.contains("for (long long i = 0"));
    }

    #[test]
    fn test_render_parallel_for_with_reduction() {
        let mut renderer = OpenMPRenderer::new();
        let body = AstNode::Const(Literal::I64(0));
        let parallel = ParallelInfo {
            is_parallel: true,
            kind: ParallelKind::OpenMP,
            reductions: vec![("sum".to_string(), ReductionOp::Add)],
        };

        let code = renderer.render_range(
            "i",
            &AstNode::Const(Literal::I64(0)),
            &AstNode::Var("n".to_string()),
            &AstNode::Const(Literal::I64(1)),
            &body,
            &parallel,
        );

        assert!(code.contains("#pragma omp parallel for"));
        assert!(code.contains("reduction(+:sum)"));
    }

    #[test]
    fn test_render_sequential_for() {
        let mut renderer = OpenMPRenderer::new();
        let body = AstNode::Const(Literal::I64(0));
        let parallel = ParallelInfo::default();

        let code = renderer.render_range(
            "i",
            &AstNode::Const(Literal::I64(0)),
            &AstNode::Var("n".to_string()),
            &AstNode::Const(Literal::I64(1)),
            &body,
            &parallel,
        );

        assert!(!code.contains("#pragma omp"));
        assert!(code.contains("for (long long i = 0"));
    }

    #[test]
    fn test_render_barrier() {
        let renderer = OpenMPRenderer::new();
        let barrier = renderer.render_barrier_backend();
        assert!(barrier.contains("#pragma omp barrier"));
    }

    #[test]
    fn test_openmpcode_display() {
        let code = OpenMPCode::new("int main() { return 0; }".to_string());
        assert_eq!(format!("{}", code), "int main() { return 0; }");
    }
}
