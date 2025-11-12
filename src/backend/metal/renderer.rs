use crate::ast::{AstNode, DType, FunctionKind, Mutability, VarDecl, VarKind};
use crate::backend::Renderer;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::metal::MetalCode;
use log::{info, trace};

/// Metal Shading Language用のレンダラー
pub struct MetalRenderer {
    indent_level: usize,
}

impl MetalRenderer {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// 関数を描画（パブリックメソッドとして維持）
    /// プログラム全体を描画
    pub fn render_program(&mut self, program: &AstNode) -> MetalCode {
        self.render_program_with_signature(program, crate::backend::KernelSignature::empty())
    }

    /// シグネチャ付きでプログラムをレンダリング
    pub fn render_program_with_signature(
        &mut self,
        program: &AstNode,
        signature: crate::backend::KernelSignature,
    ) -> MetalCode {
        if let AstNode::Program {
            functions,
            entry_point,
        } = program
        {
            info!(
                "Rendering Metal program: {} with {} functions",
                entry_point,
                functions.len()
            );

            let result = CLikeRenderer::render_program_clike(self, program);

            info!("Metal program rendering completed ({} bytes)", result.len());
            trace!("Generated Metal code:\n{}", result);

            MetalCode::with_signature(result, signature)
        } else {
            panic!("Expected AstNode::Program");
        }
    }
}

// CLikeRendererトレイトの実装
impl CLikeRenderer for MetalRenderer {
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
            DType::Usize => "uint".to_string(),
            DType::Ptr(inner) => format!("device {}*", self.render_dtype_backend(inner)),
            DType::Vec(inner, size) => {
                let base = self.render_dtype_backend(inner);
                format!("{}{}", base, size)
            }
            DType::Tuple(types) => {
                if types.is_empty() {
                    "void".to_string()
                } else {
                    // Metalはタプル型を直接サポートしないので構造体として表現
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
        "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string()
    }

    fn render_header(&self) -> String {
        "#include <metal_stdlib>\nusing namespace metal;\n\n".to_string()
    }

    fn render_function_qualifier(&self, func_kind: &FunctionKind) -> String {
        match func_kind {
            FunctionKind::Kernel(_) => "kernel".to_string(),
            FunctionKind::Normal => String::new(),
        }
    }

    fn render_param_attribute(&self, param: &VarDecl, is_kernel: bool) -> String {
        let type_str = self.render_dtype_backend(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        if is_kernel {
            match &param.kind {
                VarKind::Normal => {
                    format!("{}{} {}", mut_str, type_str, param.name)
                }
                VarKind::ThreadId(_) => {
                    format!("uint {} [[thread_position_in_grid]]", param.name)
                }
                VarKind::GroupId(_) => {
                    format!("uint {} [[threadgroup_position_in_grid]]", param.name)
                }
                VarKind::GroupSize(_) => {
                    format!("uint {} [[threads_per_threadgroup]]", param.name)
                }
                VarKind::GridSize(_) => {
                    format!("uint {} [[threads_per_grid]]", param.name)
                }
            }
        } else {
            format!("{}{} {}", mut_str, type_str, param.name)
        }
    }

    fn render_thread_var_declarations(&self, _params: &[VarDecl], _indent: &str) -> String {
        // Metalではスレッド変数はパラメータ属性として宣言されるので、ここでは何もしない
        String::new()
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        match name {
            "max" => format!("max({})", args.join(", ")),
            "sqrt" => format!("sqrt({})", args.join(", ")),
            "log2" => format!("log2({})", args.join(", ")),
            "exp2" => format!("exp2({})", args.join(", ")),
            "sin" => format!("sin({})", args.join(", ")),
            _ => format!("{}({})", name, args.join(", ")),
        }
    }
}

impl Default for MetalRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for MetalRenderer {
    type CodeRepr = MetalCode;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut renderer = Self::new();
        renderer.render_program(program)
    }

    fn is_available(&self) -> bool {
        // Metalは常に利用可能として扱う（実際のコンパイルは別の問題）
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::{AstNode, Literal, Scope};
    use crate::backend::c_like::CLikeRenderer;

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_render_literal() {
        let renderer = MetalRenderer::new();
        assert_eq!(renderer.render_literal(&Literal::F32(3.14)), "3.14f");
        assert_eq!(renderer.render_literal(&Literal::Int(42)), "42");
    }

    #[test]
    fn test_render_dtype() {
        let renderer = MetalRenderer::new();
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::Isize), "int");
        assert_eq!(renderer.render_dtype_backend(&DType::Usize), "uint");
        assert_eq!(
            renderer.render_dtype_backend(&DType::F32.to_ptr()),
            "device float*"
        );
        assert_eq!(
            renderer.render_dtype_backend(&DType::F32.to_vec(4)),
            "float4"
        );
    }

    #[test]
    fn test_render_binary_ops() {
        let renderer = MetalRenderer::new();
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());

        let add = a.clone() + b.clone();
        assert_eq!(renderer.render_expr(&add), "(1f + 2f)");

        let mul = a.clone() * b.clone();
        assert_eq!(renderer.render_expr(&mul), "(1f * 2f)");
    }

    #[test]
    fn test_render_math_funcs() {
        let renderer = MetalRenderer::new();
        let x = AstNode::Const(4.0f32.into());

        assert_eq!(renderer.render_expr(&sqrt(x.clone())), "sqrt(4f)");
        assert_eq!(renderer.render_expr(&sin(x.clone())), "sin(4f)");
        assert_eq!(renderer.render_expr(&log2(x.clone())), "log2(4f)");
    }

    #[test]
    fn test_render_barrier() {
        let mut renderer = MetalRenderer::new();
        let barrier_stmt = renderer.render_statement(&barrier());
        assert!(barrier_stmt.contains("threadgroup_barrier"));
    }

    #[test]
    fn test_render_simple_kernel() {
        // 簡単なカーネル: output[tid] = input[tid] * 2.0
        let params = vec![
            VarDecl {
                name: "tid".to_string(),
                dtype: DType::Usize,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(0),
                initial_value: None,
            },
            VarDecl {
                name: "input".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
                initial_value: None,
            },
            VarDecl {
                name: "output".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
                initial_value: None,
            },
        ];

        let body_statements = vec![store(
            var("output"),
            var("tid"),
            load(var("input"), var("tid"), DType::F32) * AstNode::Const(2.0f32.into()),
        )];

        use crate::ast::Scope;
        let func = AstNode::Function {
            kind: FunctionKind::Kernel(1),
            name: Some("scale_kernel".to_string()),
            params,
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: body_statements,
                scope: Box::new(Scope::new()),
            }),
        };

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_function_node(&func);

        // 基本的な構造をチェック
        assert!(code.contains("kernel"));
        assert!(code.contains("void scale_kernel"));
        assert!(code.contains("thread_position_in_grid"));
        assert!(code.contains("device float*"));
        assert!(code.contains("output[tid] = (input[tid] * 2f)"));
    }

    #[test]
    fn test_render_program() {
        use crate::ast::Scope;

        // 簡単な関数: double(x) = x * 2
        let double_params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
            initial_value: None,
        }];

        let double_func = AstNode::Function {
            kind: FunctionKind::Normal,
            name: Some("double".to_string()),
            params: double_params,
            return_type: DType::F32,
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Return {
                    value: Box::new(var("x") * AstNode::Const(2.0f32.into())),
                }],
                scope: Box::new(Scope::new()),
            }),
        };

        // メイン関数
        let main_func = AstNode::Function {
            kind: FunctionKind::Normal,
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::F32,
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Call {
                    name: "double".to_string(),
                    args: vec![AstNode::Const(5.0f32.into())],
                }],
                scope: Box::new(Scope::new()),
            }),
        };

        let program = AstNode::Program {
            functions: vec![double_func, main_func],
            entry_point: "main".to_string(),
        };

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_program(&program);

        // ヘッダーとインクルードをチェック
        assert!(code.contains("#include <metal_stdlib>"));
        assert!(code.contains("using namespace metal;"));

        // 関数定義をチェック
        assert!(code.contains("float double("));
        assert!(code.contains("float main("));
        assert!(code.contains("return (x * 2f)"));
        assert!(code.contains("double(5f)"));
    }

    #[test]
    fn test_render_loop_with_barrier() {
        // ループとバリアを含むカーネル
        let mut loop_scope = Scope::new();
        loop_scope
            .declare("i".to_string(), DType::Usize, Mutability::Immutable, None)
            .unwrap();

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const((0 as isize).into())),
            step: Box::new(AstNode::Const((1 as isize).into())),
            stop: Box::new(AstNode::Const((10 as isize).into())),
            body: Box::new(AstNode::Block {
                statements: vec![
                    store(
                        var("shared"),
                        var("i"),
                        load(var("input"), var("i"), DType::F32),
                    ),
                    barrier(),
                    store(
                        var("output"),
                        var("i"),
                        load(var("shared"), var("i"), DType::F32),
                    ),
                ],
                scope: Box::new(loop_scope),
            }),
        };

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_statement(&loop_node);

        assert!(code.contains("for (uint i = 0; i < 10; i += 1)"));
        assert!(code.contains("shared[i] = input[i]"));
        assert!(code.contains("threadgroup_barrier"));
        assert!(code.contains("output[i] = shared[i]"));
    }
}
