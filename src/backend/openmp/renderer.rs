use crate::ast::{AstNode, DType, FunctionKind, Mutability, VarDecl, VarKind};
use crate::backend::Renderer;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::openmp::CCode;

/// C言語とOpenMP用のレンダラー
#[derive(Debug, Clone)]
pub struct CRenderer {
    indent_level: usize,
}

impl CRenderer {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// Programをレンダリング
    pub fn render_program(&mut self, program: &AstNode) -> CCode {
        let code = CLikeRenderer::render_program_clike(self, program);
        CCode::new(code)
    }
}

// CLikeRendererトレイトの実装
impl CLikeRenderer for CRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "int".to_string(),
            DType::Usize => "unsigned int".to_string(),
            DType::Ptr(inner) => format!("{}*", self.render_dtype_backend(inner)),
            DType::Vec(inner, size) => {
                // C言語ではベクトル型をサポートしていないので、配列として表現
                format!("{}[{}]", self.render_dtype_backend(inner), size)
            }
            DType::Tuple(types) => {
                if types.is_empty() {
                    "void".to_string()
                } else {
                    // タプル型は構造体として表現
                    format!("tuple_{}", types.len())
                }
            }
            DType::Unknown => {
                panic!(
                    "Type inference failed: DType::Unknown should not appear in code generation. This indicates a bug in type inference."
                )
            }
        }
    }

    fn render_barrier_backend(&self) -> String {
        "#pragma omp barrier".to_string()
    }

    fn render_header(&self) -> String {
        "#include <math.h>\n#include <omp.h>\n#include <stdint.h>\n\n".to_string()
    }

    fn render_function_qualifier(&self, _func_kind: &FunctionKind) -> String {
        // C言語/OpenMPには関数修飾子がない
        String::new()
    }

    fn render_param_attribute(&self, param: &VarDecl, _is_kernel: bool) -> String {
        let type_str = self.render_dtype_backend(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        // ThreadId, GroupIdなどは関数内で取得するため、パラメータには含めない
        match &param.kind {
            VarKind::Normal => {
                format!("{}{} {}", mut_str, type_str, param.name)
            }
            VarKind::ThreadId(_)
            | VarKind::GroupId(_)
            | VarKind::GroupSize(_)
            | VarKind::GridSize(_) => {
                // これらは関数内で宣言するため、ここでは空文字を返す
                String::new()
            }
        }
    }

    fn render_thread_var_declarations(&self, params: &[VarDecl], indent: &str) -> String {
        let mut result = String::new();
        for param in params {
            match &param.kind {
                VarKind::ThreadId(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = omp_get_thread_num();\n",
                        indent, param.name
                    ));
                }
                VarKind::GroupId(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = 0; // group_id not supported\n",
                        indent, param.name
                    ));
                }
                VarKind::GroupSize(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = omp_get_num_threads();\n",
                        indent, param.name
                    ));
                }
                VarKind::GridSize(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = 0; // grid_size not supported\n",
                        indent, param.name
                    ));
                }
                VarKind::Normal => {}
            }
        }
        result
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        match name {
            "max" => format!("fmaxf({})", args.join(", ")),
            "sqrt" => format!("sqrtf({})", args.join(", ")),
            "log2" => format!("log2f({})", args.join(", ")),
            "exp2" => format!("exp2f({})", args.join(", ")),
            "sin" => format!("sinf({})", args.join(", ")),
            _ => format!("{}({})", name, args.join(", ")),
        }
    }
}

impl Default for CRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for CRenderer {
    type CodeRepr = CCode;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut renderer = Self::new();
        renderer.render_program(program)
    }

    fn is_available(&self) -> bool {
        // C/OpenMPは常に利用可能（コンパイラがあれば）
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, DType, Literal, helper::*};
    use crate::backend::c_like::CLikeRenderer;

    #[test]
    fn test_render_dtype() {
        let renderer = CRenderer::new();
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::Isize), "int");
        assert_eq!(renderer.render_dtype_backend(&DType::Usize), "unsigned int");
        assert_eq!(
            renderer.render_dtype_backend(&DType::Ptr(Box::new(DType::F32))),
            "float*"
        );
    }

    #[test]
    fn test_render_literal() {
        let renderer = CRenderer::new();
        assert_eq!(renderer.render_literal(&Literal::F32(1.5)), "1.5f");
        assert_eq!(renderer.render_literal(&Literal::Isize(42)), "42");
        assert_eq!(renderer.render_literal(&Literal::Usize(10)), "10u");
    }

    #[test]
    fn test_render_expr() {
        let renderer = CRenderer::new();

        let add_expr = AstNode::Add(Box::new(var("a")), Box::new(var("b")));
        assert_eq!(renderer.render_expr(&add_expr), "(a + b)");

        let mul_expr = AstNode::Mul(Box::new(var("x")), Box::new(var("y")));
        assert_eq!(renderer.render_expr(&mul_expr), "(x * y)");
    }

    #[test]
    fn test_render_simple_program() {
        use crate::ast::Scope;

        let func = AstNode::Function {
            kind: FunctionKind::Normal,
            name: Some("test_func".to_string()),
            params: vec![VarDecl {
                name: "x".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
                initial_value: None,
            }],
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: vec![store(
                    var("x"),
                    AstNode::Const(Literal::Usize(0)),
                    AstNode::Const(Literal::F32(1.0)),
                )],
                scope: Box::new(Scope::new()),
            }),
        };

        let program = AstNode::Program {
            functions: vec![func],
            entry_point: "test_func".to_string(),
        };

        let renderer = CRenderer::new();
        let code = renderer.render(&program);

        assert!(code.contains("#include <math.h>"));
        assert!(code.contains("#include <omp.h>"));
        assert!(code.contains("void test_func("));
        assert!(code.contains("float* x"));
    }
}
