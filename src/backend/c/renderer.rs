use crate::ast::{AstNode, DType, Mutability, VarDecl, VarKind};
use crate::backend::Renderer;
use crate::backend::c::{CCode, LIBLOADING_WRAPPER_NAME};
use crate::backend::c_like::CLikeRenderer;

/// C言語レンダラー（シングルスレッド実行専用）
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
        CCode::new(CLikeRenderer::render_program_clike(self, program))
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
            DType::Bool => "unsigned char".to_string(), // boolはu8として表現
            DType::F32 => "float".to_string(),
            DType::Int => "int".to_string(),
            DType::Ptr(inner) => format!("{}*", self.render_dtype_backend(inner)),
            DType::Vec(inner, size) => {
                // GCC/Clangのベクトル拡張を使用
                let base = self.render_dtype_backend(inner);
                format!("{}{}", base, size)
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
        // シングルスレッド実行のため、バリアは不要（空文字列を返す）
        String::new()
    }

    fn render_header(&self) -> String {
        let mut header = String::from("#include <math.h>\n");
        header.push_str("#include <stdint.h>\n");
        header.push_str("#include <stdlib.h>\n\n");

        // GCC/Clangのベクトル拡張を使用したベクトル型の定義
        header.push_str("// SIMD vector types using GCC/Clang vector extensions\n");
        header.push_str("typedef float float2 __attribute__((vector_size(8)));\n");
        header.push_str("typedef float float4 __attribute__((vector_size(16)));\n");
        header.push_str("typedef float float8 __attribute__((vector_size(32)));\n");
        header.push_str("typedef int int2 __attribute__((vector_size(8)));\n");
        header.push_str("typedef int int4 __attribute__((vector_size(16)));\n");
        header.push_str("typedef int int8 __attribute__((vector_size(32)));\n");
        header.push('\n');

        header
    }

    fn render_function_qualifier(&self, _is_kernel: bool) -> String {
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
                        "{}int {} = omp_get_thread_num();\n",
                        indent, param.name
                    ));
                }
                VarKind::GroupId(_) => {
                    result.push_str(&format!(
                        "{}int {} = 0; // group_id not supported\n",
                        indent, param.name
                    ));
                }
                VarKind::GroupSize(_) => {
                    result.push_str(&format!(
                        "{}int {} = omp_get_num_threads();\n",
                        indent, param.name
                    ));
                }
                VarKind::GridSize(_) => {
                    result.push_str(&format!(
                        "{}int {} = 0; // grid_size not supported\n",
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

    fn render_vector_load(&self, ptr_expr: &str, offset_expr: &str, dtype: &str) -> String {
        // C言語のキャスト構文を使用
        format!("*({}*)(&{}[{}])", dtype, ptr_expr, offset_expr)
    }

    fn libloading_wrapper_name(&self) -> &'static str {
        LIBLOADING_WRAPPER_NAME
    }

    fn render_libloading_wrapper(&self, entry_func: &AstNode, entry_point: &str) -> String {
        if let AstNode::Function { params, .. } = entry_func {
            let mut result = String::new();

            result.push_str("// === libloading Wrapper ===\n");
            result.push_str(&format!(
                "void {}(void** buffers) {{\n",
                self.libloading_wrapper_name()
            ));

            // エントリーポイント関数の呼び出しを生成
            let mut call_args = Vec::new();
            let mut buffer_idx = 0;

            for param in params {
                // ThreadId等の特殊な変数はスキップ（パラメータにならない）
                match &param.kind {
                    VarKind::Normal => {
                        // パラメータの型に基づいてキャストを生成
                        let type_str = self.render_dtype_backend(&param.dtype);
                        let mut_str = match param.mutability {
                            Mutability::Immutable => "const ",
                            Mutability::Mutable => "",
                        };
                        call_args.push(format!(
                            "({}{})(buffers[{}])",
                            mut_str, type_str, buffer_idx
                        ));
                        buffer_idx += 1;
                    }
                    VarKind::ThreadId(_)
                    | VarKind::GroupId(_)
                    | VarKind::GroupSize(_)
                    | VarKind::GridSize(_) => {
                        // これらはパラメータに含まれないのでスキップ
                    }
                }
            }

            result.push_str(&format!("    {}({});\n", entry_point, call_args.join(", ")));
            result.push_str("}\n");

            result
        } else {
            String::new()
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
        assert_eq!(renderer.render_dtype_backend(&DType::Bool), "unsigned char");
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::Int), "int");
        assert_eq!(
            renderer.render_dtype_backend(&DType::Ptr(Box::new(DType::F32))),
            "float*"
        );
    }

    #[test]
    fn test_render_literal() {
        let renderer = CRenderer::new();
        assert_eq!(renderer.render_literal(&Literal::Bool(true)), "1");
        assert_eq!(renderer.render_literal(&Literal::Bool(false)), "0");
        assert_eq!(renderer.render_literal(&Literal::F32(1.5)), "1.5f");
        assert_eq!(renderer.render_literal(&Literal::Int(42)), "42");
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
        use crate::backend::c::LIBLOADING_WRAPPER_NAME;

        let func = AstNode::Function {
            name: Some("test_func".to_string()),
            params: vec![VarDecl {
                name: "x".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: vec![store(
                    var("x"),
                    AstNode::Const(Literal::Int(0)),
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
        assert!(code.contains("void test_func("));
        assert!(code.contains("float* x"));

        // libloading用のラッパー関数が生成されていることを確認
        assert!(
            code.contains(&format!("void {}(void** buffers)", LIBLOADING_WRAPPER_NAME)),
            "libloading wrapper should be generated"
        );
        assert!(
            code.contains("test_func((float*)(buffers[0]))"),
            "wrapper should call entry point with correct cast"
        );
    }

    #[test]
    fn test_render_simd_vector_type() {
        let renderer = CRenderer::new();

        // ベクトル型のレンダリング
        let vec_type = DType::Vec(Box::new(DType::F32), 4);
        assert_eq!(renderer.render_dtype_backend(&vec_type), "float4");

        // ヘッダーにベクトル型の定義が含まれることを確認
        let header = renderer.render_header();
        assert!(header.contains("typedef float float4"));
        assert!(header.contains("__attribute__((vector_size(16)))"));
    }

    #[test]
    fn test_render_vector_load() {
        let renderer = CRenderer::new();

        // ベクトルロードのレンダリング
        let load_code = renderer.render_vector_load("input", "i", "float4");
        assert_eq!(load_code, "*(float4*)(&input[i])");
    }

    #[test]
    fn test_libloading_wrapper_multiple_params() {
        use crate::ast::Scope;
        use crate::backend::c::LIBLOADING_WRAPPER_NAME;

        let func = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![
                VarDecl {
                    name: "input0".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "input1".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: vec![],
                scope: Box::new(Scope::new()),
            }),
        };

        let program = AstNode::Program {
            functions: vec![func],
            entry_point: "main".to_string(),
        };

        let renderer = CRenderer::new();
        let code = renderer.render(&program);

        // ラッパー関数のシグネチャを確認
        assert!(code.contains(&format!("void {}(void** buffers)", LIBLOADING_WRAPPER_NAME)));

        // 各パラメータが正しくキャストされていることを確認
        assert!(code.contains("(const float*)(buffers[0])"));
        assert!(code.contains("(const float*)(buffers[1])"));
        assert!(code.contains("(float*)(buffers[2])"));

        // 呼び出しが正しい形式であることを確認
        assert!(code.contains(
            "main((const float*)(buffers[0]), (const float*)(buffers[1]), (float*)(buffers[2]))"
        ));
    }
}
