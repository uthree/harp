//! Pure C language renderer for Harp
//!
//! This module provides a renderer that generates pure C code without
//! any parallelization or external library dependencies (except standard C library).
//!
//! The generated code is suitable for:
//! - CPU execution with sequential processing
//! - Embedding in other C/C++ projects
//! - Portability across different platforms

use harp_core::ast::{AstNode, DType, VarDecl};
use harp_core::backend::Renderer;
use harp_core::backend::renderer::CLikeRenderer;

/// Pure C source code representation
///
/// Uses newtype pattern to provide type safety and prevent
/// accidentally mixing with other backend code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CCode(String);

impl CCode {
    /// Create new CCode
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

impl From<String> for CCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<CCode> for String {
    fn from(code: CCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for CCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for CCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Pure C code renderer
///
/// Generates portable C99 code without parallelization or external dependencies.
/// All kernel functions are converted to regular C functions.
#[derive(Debug, Clone, Default)]
pub struct CRenderer {
    indent_level: usize,
}

impl CRenderer {
    /// Create a new CRenderer
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }
}

impl Renderer for CRenderer {
    type CodeRepr = CCode;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut r = self.clone();
        CCode::new(r.render_program_clike(program))
    }

    fn is_available(&self) -> bool {
        true // Pure C is always available
    }
}

impl CLikeRenderer for CRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        match dtype {
            DType::Bool => "unsigned char".to_string(),
            DType::I8 => "signed char".to_string(),
            DType::I16 => "short".to_string(),
            DType::I32 => "int".to_string(),
            DType::I64 => "long long".to_string(),
            DType::U8 => "unsigned char".to_string(),
            DType::U16 => "unsigned short".to_string(),
            DType::U32 => "unsigned int".to_string(),
            DType::U64 => "unsigned long long".to_string(),
            DType::F32 => "float".to_string(),
            DType::F64 => "double".to_string(),
            DType::Int => "long long".to_string(), // Index type: 64-bit for CPU
            DType::Ptr(inner) => format!("{}*", self.render_dtype_backend(inner)),
            DType::Vec(inner, size) => format!("{}[{}]", self.render_dtype_backend(inner), size),
            DType::Tuple(types) => {
                if types.is_empty() {
                    "void".to_string()
                } else {
                    // For tuples, we use a comment placeholder
                    format!("/* tuple_{} */", types.len())
                }
            }
            DType::Unknown => "/* unknown */".to_string(),
        }
    }

    fn render_barrier_backend(&self) -> String {
        // No barrier in sequential C code
        "/* barrier (no-op in sequential C) */".to_string()
    }

    fn render_header(&self) -> String {
        r#"/* Generated C code by Harp */
/* This code is pure C99 without parallelization */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>

"#
        .to_string()
    }

    fn render_function_qualifier(&self, _is_kernel: bool) -> String {
        // In pure C, kernels are just regular functions
        String::new()
    }

    fn render_param_attribute(&self, param: &VarDecl, _is_kernel: bool) -> String {
        // Thread ID parameters become regular function parameters
        let type_str = self.render_dtype_backend(&param.dtype);
        format!("{} {}", type_str, param.name)
    }

    fn render_thread_var_declarations(&self, _params: &[VarDecl], _indent: &str) -> String {
        // No thread-local variables in sequential C
        String::new()
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        // Use standard C math functions
        match name {
            "max" => {
                // C doesn't have max for floats, use fmaxf
                format!("fmaxf({}, {})", args[0], args[1])
            }
            "min" => {
                format!("fminf({}, {})", args[0], args[1])
            }
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
        // In sequential C, atomic operations are just regular operations
        format!(
            "/* atomic_add (sequential) */ ({}[{}] += {})",
            ptr, offset, value
        )
    }

    fn render_atomic_max(&self, ptr: &str, offset: &str, value: &str, _dtype: &DType) -> String {
        // In sequential C, atomic max is just a comparison and assignment
        format!(
            "/* atomic_max (sequential) */ ({{float __tmp = {}[{}]; {}[{}] = __tmp > {} ? __tmp : {}; __tmp;}})",
            ptr, offset, ptr, offset, value, value
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harp_core::ast::helper::*;
    use harp_core::ast::{AstNode, Literal};
    use harp_core::backend::renderer::CLikeRenderer;

    #[test]
    fn test_render_header() {
        let renderer = CRenderer::new();
        let header = renderer.render_header();
        assert!(header.contains("#include <math.h>"));
        assert!(header.contains("#include <stdlib.h>"));
        assert!(header.contains("C99"));
    }

    #[test]
    fn test_render_dtype() {
        let renderer = CRenderer::new();
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::F64), "double");
        assert_eq!(renderer.render_dtype_backend(&DType::I32), "int");
        assert_eq!(renderer.render_dtype_backend(&DType::I64), "long long");
        assert_eq!(
            renderer.render_dtype_backend(&DType::Ptr(Box::new(DType::F32))),
            "float*"
        );
    }

    #[test]
    fn test_render_math_functions() {
        let renderer = CRenderer::new();
        assert_eq!(
            renderer.render_math_func("sqrt", &["x".to_string()]),
            "sqrtf(x)"
        );
        assert_eq!(
            renderer.render_math_func("max", &["a".to_string(), "b".to_string()]),
            "fmaxf(a, b)"
        );
    }

    #[test]
    fn test_render_binary_ops() {
        let renderer = CRenderer::new();
        let a = AstNode::Const(Literal::F32(1.0));
        let b = AstNode::Const(Literal::F32(2.0));

        let add = a.clone() + b.clone();
        assert_eq!(renderer.render_expr(&add), "(1.0f + 2.0f)");

        let mul = a.clone() * b.clone();
        assert_eq!(renderer.render_expr(&mul), "(1.0f * 2.0f)");
    }

    #[test]
    fn test_render_math_expressions() {
        let renderer = CRenderer::new();
        let x = AstNode::Const(Literal::F32(4.0));

        assert_eq!(renderer.render_expr(&sqrt(x.clone())), "sqrtf(4.0f)");
        assert_eq!(renderer.render_expr(&sin(x.clone())), "sinf(4.0f)");
        assert_eq!(renderer.render_expr(&log2(x.clone())), "log2f(4.0f)");
    }

    #[test]
    fn test_render_barrier() {
        let mut renderer = CRenderer::new();
        let barrier_stmt = renderer.render_statement(&barrier());
        assert!(barrier_stmt.contains("barrier"));
        assert!(barrier_stmt.contains("no-op"));
    }

    #[test]
    fn test_ccode_display() {
        let code = CCode::new("int main() { return 0; }".to_string());
        assert_eq!(format!("{}", code), "int main() { return 0; }");
    }

    #[test]
    fn test_ccode_conversions() {
        let code = CCode::new("test code".to_string());
        assert_eq!(code.as_str(), "test code");
        assert_eq!(code.len(), 9);
        assert!(!code.is_empty());
        assert!(code.contains("test"));

        let string: String = code.into();
        assert_eq!(string, "test code");
    }
}
