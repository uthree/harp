use crate::ast::{AstNode, DType, Mutability, VarDecl, VarKind};
use crate::backend::Renderer;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::metal::{LIBLOADING_WRAPPER_NAME, MetalCode};

// ============================================================
// Metal共通のレンダリングヘルパー関数
// MetalRendererとMetalKernelRendererの両方で使用
// ============================================================

/// Metal用のDType変換
fn render_dtype_metal(dtype: &DType) -> String {
    match dtype {
        DType::Bool => "uchar".to_string(),
        DType::F32 => "float".to_string(),
        DType::Int => "int".to_string(),
        DType::Ptr(inner) => format!("device {}*", render_dtype_metal(inner)),
        DType::Vec(inner, size) => {
            let base = render_dtype_metal(inner);
            format!("{}{}", base, size)
        }
        DType::Tuple(types) => {
            if types.is_empty() {
                "void".to_string()
            } else {
                format!("tuple_{}", types.len())
            }
        }
        DType::Unknown => {
            panic!("Type inference failed: DType::Unknown should not appear in code generation")
        }
    }
}

/// Metal用のバリア命令
fn render_barrier_metal() -> String {
    "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string()
}

/// Metal Shading Languageヘッダー
fn render_header_metal() -> String {
    "#include <metal_stdlib>\nusing namespace metal;\n\n".to_string()
}

/// Metal用の関数修飾子
fn render_function_qualifier_metal(is_kernel: bool) -> String {
    if is_kernel {
        "kernel".to_string()
    } else {
        String::new()
    }
}

/// Metal用のパラメータ属性
fn render_param_attribute_metal(param: &VarDecl, is_kernel: bool) -> String {
    let type_str = render_dtype_metal(&param.dtype);
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
            VarKind::LocalId(_) => {
                format!("uint {} [[thread_position_in_threadgroup]]", param.name)
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

/// Metal用の数学関数
fn render_math_func_metal(name: &str, args: &[String]) -> String {
    match name {
        "max" => format!("max({})", args.join(", ")),
        "sqrt" => format!("sqrt({})", args.join(", ")),
        "log2" => format!("log2({})", args.join(", ")),
        "exp2" => format!("exp2({})", args.join(", ")),
        "sin" => format!("sin({})", args.join(", ")),
        _ => format!("{}({})", name, args.join(", ")),
    }
}

// ============================================================
// MetalRenderer - Objective-C++ + Metal API ホストコード生成
// ============================================================

/// Metal Shading Language用のレンダラー（C++ラッパー方式）
#[derive(Clone)]
pub struct MetalRenderer {
    indent_level: usize,
}

impl MetalRenderer {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// プログラム全体をObjective-C++ + Metal APIコードとして描画
    pub fn render_program(&mut self, program: &AstNode) -> MetalCode {
        self.render_program_with_signature(program, crate::backend::KernelSignature::empty())
    }

    /// シグネチャ付きでプログラムをレンダリング
    ///
    /// カーネル関数群をMetal Shading Language + ホストコードとして出力します。
    /// カーネルの実行順序はホスト側（CompiledProgram）で管理されます。
    ///
    /// Note: signatureパラメータは互換性のために残されていますが、
    /// Code型がsignatureを保持しなくなったため無視されます。
    pub fn render_program_with_signature(
        &mut self,
        program: &AstNode,
        _signature: crate::backend::KernelSignature,
    ) -> MetalCode {
        if let AstNode::Program { functions } = program {
            let mut code = String::new();

            // 1. ヘッダー
            code.push_str(&self.render_objcpp_header());
            code.push_str("\n\n");

            // 2. Metal Shading Languageカーネルを文字列リテラルとして生成
            let mut kernel_renderer = MetalKernelRenderer::new();
            let kernel_source = kernel_renderer.render_kernel_source(functions);

            code.push_str("// Metal Shading Language kernel source\n");
            code.push_str("const char* METAL_KERNEL_SOURCE = R\"(\n");
            code.push_str(&kernel_source);
            code.push_str(")\";\n\n");

            // 3. libloading用のホストコード生成（最初のカーネルをデフォルトとして使用）
            let first_kernel_name = functions
                .iter()
                .find_map(|f| match f {
                    AstNode::Kernel { name: Some(n), .. } => Some(n.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "unknown_kernel".to_string());
            code.push_str(&self.generate_host_code(&first_kernel_name, functions));

            MetalCode::new(code)
        } else {
            panic!("Expected AstNode::Program");
        }
    }

    /// Objective-C++ヘッダーを生成
    fn render_objcpp_header(&self) -> String {
        let mut header = String::new();
        header.push_str("#include <Metal/Metal.h>\n");
        header.push_str("#include <Foundation/Foundation.h>\n");
        header.push_str("#include <stdio.h>\n");
        header.push_str("#include <stdlib.h>\n");
        header
    }

    /// Metal APIを使ったホストコードを生成
    fn generate_host_code(&self, entry_point: &str, functions: &[AstNode]) -> String {
        // エントリーポイント関数のパラメータを取得
        let entry_func = functions.iter().find(
            |f| matches!(f, AstNode::Function { name: Some(name), .. } if name == entry_point),
        );

        let buffer_count = if let Some(AstNode::Function { params, .. }) = entry_func {
            params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::Normal))
                .count()
        } else {
            0
        };

        let mut code = String::new();

        code.push_str("// === Metal API Host Code ===\n");
        code.push_str("extern \"C\" {\n");
        code.push_str(&format!(
            "void {}(void** buffers) {{\n",
            LIBLOADING_WRAPPER_NAME
        ));
        code.push_str("    @autoreleasepool {\n");
        code.push_str("        // Initialize Metal device\n");
        code.push_str("        id<MTLDevice> device = MTLCreateSystemDefaultDevice();\n");
        code.push_str("        if (!device) {\n");
        code.push_str("            fprintf(stderr, \"Failed to create Metal device\\n\");\n");
        code.push_str("            return;\n");
        code.push_str("        }\n\n");

        code.push_str("        // Create command queue\n");
        code.push_str("        id<MTLCommandQueue> commandQueue = [device newCommandQueue];\n\n");

        code.push_str("        // Compile kernel source\n");
        code.push_str("        NSError* error = nil;\n");
        code.push_str("        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];\n");
        code.push_str("        id<MTLLibrary> library = [device newLibraryWithSource:@(METAL_KERNEL_SOURCE)\n");
        code.push_str("                                                      options:options\n");
        code.push_str("                                                        error:&error];\n");
        code.push_str("        if (!library) {\n");
        code.push_str("            NSLog(@\"Failed to compile Metal library: %@\", error);\n");
        code.push_str("            return;\n");
        code.push_str("        }\n\n");

        code.push_str(&format!(
            "        // Get kernel function: {}\n",
            entry_point
        ));
        code.push_str(&format!(
            "        id<MTLFunction> function = [library newFunctionWithName:@\"{}\"];\n",
            entry_point
        ));
        code.push_str("        if (!function) {\n");
        code.push_str(&format!(
            "            fprintf(stderr, \"Failed to find kernel function '{}'\\n\");\n",
            entry_point
        ));
        code.push_str("            return;\n");
        code.push_str("        }\n\n");

        code.push_str("        // Create pipeline state\n");
        code.push_str("        id<MTLComputePipelineState> pipelineState =\n");
        code.push_str(
            "            [device newComputePipelineStateWithFunction:function error:&error];\n",
        );
        code.push_str("        if (!pipelineState) {\n");
        code.push_str("            NSLog(@\"Failed to create pipeline state: %@\", error);\n");
        code.push_str("            return;\n");
        code.push_str("        }\n\n");

        code.push_str("        // Create Metal buffers from host buffers\n");
        code.push_str(&format!(
            "        id<MTLBuffer> metalBuffers[{}];\n",
            buffer_count
        ));
        code.push_str(&format!(
            "        for (int i = 0; i < {}; i++) {{\n",
            buffer_count
        ));
        code.push_str("            // TODO: Get actual buffer size\n");
        code.push_str("            size_t bufferSize = 1024 * sizeof(float);\n");
        code.push_str("            metalBuffers[i] = [device newBufferWithBytes:buffers[i]\n");
        code.push_str("                                                  length:bufferSize\n");
        code.push_str(
            "                                                 options:MTLResourceStorageModeShared];\n",
        );
        code.push_str("        }\n\n");

        code.push_str("        // Create command buffer and encoder\n");
        code.push_str(
            "        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];\n",
        );
        code.push_str("        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];\n");
        code.push_str("        [encoder setComputePipelineState:pipelineState];\n\n");

        code.push_str("        // Bind buffers\n");
        code.push_str(&format!(
            "        for (int i = 0; i < {}; i++) {{\n",
            buffer_count
        ));
        code.push_str("            [encoder setBuffer:metalBuffers[i] offset:0 atIndex:i];\n");
        code.push_str("        }\n\n");

        code.push_str("        // Execute kernel\n");
        code.push_str("        MTLSize gridSize = MTLSizeMake(1024, 1, 1);\n");
        code.push_str(
            "        NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;\n",
        );
        code.push_str("        if (threadGroupSize > 256) threadGroupSize = 256;\n");
        code.push_str("        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);\n");
        code.push_str(
            "        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];\n",
        );
        code.push_str("        [encoder endEncoding];\n\n");

        code.push_str("        // Commit and wait\n");
        code.push_str("        [commandBuffer commit];\n");
        code.push_str("        [commandBuffer waitUntilCompleted];\n\n");

        code.push_str("        // Copy results back to host buffers\n");
        code.push_str(&format!(
            "        for (int i = 0; i < {}; i++) {{\n",
            buffer_count
        ));
        code.push_str("            void* contents = [metalBuffers[i] contents];\n");
        code.push_str("            size_t bufferSize = [metalBuffers[i] length];\n");
        code.push_str("            memcpy(buffers[i], contents, bufferSize);\n");
        code.push_str("        }\n");

        code.push_str("    }\n");
        code.push_str("}\n");
        code.push_str("}\n");

        code
    }
}

/// Metal Shading Languageカーネルをレンダリングする内部レンダラー
struct MetalKernelRenderer {
    indent_level: usize,
}

impl MetalKernelRenderer {
    fn new() -> Self {
        Self { indent_level: 0 }
    }

    fn render_kernel_source(&mut self, functions: &[AstNode]) -> String {
        let mut code = String::new();

        // Metal Shading Languageヘッダー
        code.push_str("#include <metal_stdlib>\n");
        code.push_str("using namespace metal;\n\n");

        // カーネル関数のみをレンダリング
        for func in functions {
            if matches!(func, AstNode::Kernel { .. }) {
                code.push_str(&self.render_function_node(func));
                code.push('\n');
            }
        }

        code
    }
}

// MetalKernelRenderer用のRenderer実装（内部使用のためダミー）
impl Renderer for MetalKernelRenderer {
    type CodeRepr = MetalCode;
    type Option = ();

    fn render(&self, _program: &AstNode) -> Self::CodeRepr {
        panic!("MetalKernelRenderer::render should not be called directly");
    }

    fn is_available(&self) -> bool {
        true
    }
}

// MetalKernelRenderer用のCLikeRenderer実装（共通ヘルパー関数を使用）
impl CLikeRenderer for MetalKernelRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        render_dtype_metal(dtype)
    }

    fn render_barrier_backend(&self) -> String {
        render_barrier_metal()
    }

    fn render_header(&self) -> String {
        render_header_metal()
    }

    fn render_function_qualifier(&self, is_kernel: bool) -> String {
        render_function_qualifier_metal(is_kernel)
    }

    fn render_param_attribute(&self, param: &VarDecl, is_kernel: bool) -> String {
        render_param_attribute_metal(param, is_kernel)
    }

    fn render_thread_var_declarations(&self, _params: &[VarDecl], _indent: &str) -> String {
        // Metalではスレッド変数はパラメータ属性として宣言されるので、ここでは何もしない
        String::new()
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        render_math_func_metal(name, args)
    }

    fn libloading_wrapper_name(&self) -> &'static str {
        // MetalKernelRendererはカーネルソースのみ生成するため、libloadingラッパーは生成しない
        ""
    }

    fn render_libloading_wrapper(&self, _entry_func: &AstNode, _entry_point: &str) -> String {
        // MetalKernelRendererはカーネルソースのみ生成するため、libloadingラッパーは生成しない
        String::new()
    }
}

// CLikeRendererトレイトの実装（共通ヘルパー関数を使用）
impl CLikeRenderer for MetalRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        render_dtype_metal(dtype)
    }

    fn render_barrier_backend(&self) -> String {
        render_barrier_metal()
    }

    fn render_header(&self) -> String {
        render_header_metal()
    }

    fn render_function_qualifier(&self, is_kernel: bool) -> String {
        render_function_qualifier_metal(is_kernel)
    }

    fn render_param_attribute(&self, param: &VarDecl, is_kernel: bool) -> String {
        render_param_attribute_metal(param, is_kernel)
    }

    fn render_thread_var_declarations(&self, _params: &[VarDecl], _indent: &str) -> String {
        // Metalではスレッド変数はパラメータ属性として宣言されるので、ここでは何もしない
        String::new()
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        render_math_func_metal(name, args)
    }

    fn libloading_wrapper_name(&self) -> &'static str {
        LIBLOADING_WRAPPER_NAME
    }

    fn render_libloading_wrapper(&self, entry_func: &AstNode, entry_point: &str) -> String {
        // MetalRendererは独自のrender_programを使用し、render_program_clikeを使用しないため、
        // このメソッドは直接呼ばれない。generate_host_codeで同等の処理を行っている。
        // トレイト要件を満たすためのスタブ実装。
        if let AstNode::Function { .. } = entry_func {
            self.generate_host_code(entry_point, std::slice::from_ref(entry_func))
        } else {
            String::new()
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

/// Implementation of KernelSourceRenderer for native Metal backend
#[cfg(all(feature = "native-metal", target_os = "macos"))]
impl crate::backend::execution::KernelSourceRenderer for MetalRenderer {
    fn render_kernel_source(&mut self, program: &AstNode) -> String {
        if let AstNode::Program { functions, .. } = program {
            // Use the internal MetalKernelRenderer to generate kernel-only source
            let mut kernel_renderer = MetalKernelRenderer::new();
            kernel_renderer.render_kernel_source(functions)
        } else {
            String::new()
        }
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
        assert_eq!(renderer.render_literal(&Literal::Bool(true)), "1");
        assert_eq!(renderer.render_literal(&Literal::Bool(false)), "0");
        assert_eq!(renderer.render_literal(&Literal::F32(3.14)), "3.14f");
        assert_eq!(renderer.render_literal(&Literal::Int(42)), "42");
    }

    #[test]
    fn test_render_dtype() {
        let renderer = MetalRenderer::new();
        assert_eq!(renderer.render_dtype_backend(&DType::Bool), "uchar");
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::Int), "int");
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
        assert_eq!(renderer.render_expr(&add), "(1.0f + 2.0f)");

        let mul = a.clone() * b.clone();
        assert_eq!(renderer.render_expr(&mul), "(1.0f * 2.0f)");
    }

    #[test]
    fn test_render_math_funcs() {
        let renderer = MetalRenderer::new();
        let x = AstNode::Const(4.0f32.into());

        assert_eq!(renderer.render_expr(&sqrt(x.clone())), "sqrt(4.0f)");
        assert_eq!(renderer.render_expr(&sin(x.clone())), "sin(4.0f)");
        assert_eq!(renderer.render_expr(&log2(x.clone())), "log2(4.0f)");
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
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(0),
            },
            VarDecl {
                name: "input".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            },
            VarDecl {
                name: "output".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
            },
        ];

        let body_statements = vec![store(
            var("output"),
            var("tid"),
            load(var("input"), var("tid"), DType::F32) * AstNode::Const(2.0f32.into()),
        )];

        use crate::ast::Scope;
        use crate::ast::helper::const_int;
        let one = const_int(1);
        let func = AstNode::Kernel {
            name: Some("scale_kernel".to_string()),
            params,
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: body_statements,
                scope: Box::new(Scope::new()),
            }),
            default_grid_size: [
                Box::new(one.clone()),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(64)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        };

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_function_node(&func);

        // 基本的な構造をチェック
        assert!(code.contains("kernel"));
        assert!(code.contains("void scale_kernel"));
        assert!(code.contains("thread_position_in_grid"));
        assert!(code.contains("device float*"));
        assert!(code.contains("output[tid] = (input[tid] * 2.0f)"));
    }

    #[test]
    fn test_render_program() {
        use crate::ast::Scope;
        use crate::ast::helper::const_int;
        let one = const_int(1);

        // カーネル関数を作成
        let kernel_params = vec![
            VarDecl {
                name: "tid".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(0),
            },
            VarDecl {
                name: "input".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            },
            VarDecl {
                name: "output".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
            },
        ];

        let kernel_func = AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: kernel_params,
            return_type: DType::Tuple(vec![]),
            body: Box::new(AstNode::Block {
                statements: vec![store(
                    var("output"),
                    var("tid"),
                    load(var("input"), var("tid"), DType::F32) * AstNode::Const(2.0f32.into()),
                )],
                scope: Box::new(Scope::new()),
            }),
            default_grid_size: [
                Box::new(one.clone()),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(64)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        };

        let program = AstNode::Program {
            functions: vec![kernel_func],
        };

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_program(&program);

        // Objective-C++ヘッダーをチェック
        assert!(code.contains("#include <Metal/Metal.h>"));
        assert!(code.contains("#include <Foundation/Foundation.h>"));

        // Metal Shading Languageカーネルソースがrawリテラルに埋め込まれているかチェック
        assert!(code.contains("const char* METAL_KERNEL_SOURCE = R\"("));
        assert!(code.contains("#include <metal_stdlib>"));
        assert!(code.contains("using namespace metal;"));

        // libloading用のエントリーポイントをチェック
        assert!(code.contains("extern \"C\""));
        assert!(code.contains("void __harp_metal_entry(void** buffers)"));

        // Metal API呼び出しをチェック
        assert!(code.contains("MTLCreateSystemDefaultDevice"));
        assert!(code.contains("newLibraryWithSource"));
    }

    #[test]
    fn test_render_loop_with_barrier() {
        // ループとバリアを含むカーネル
        let mut loop_scope = Scope::new();
        loop_scope
            .declare("i".to_string(), DType::Int, Mutability::Immutable)
            .unwrap();

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0_isize.into())),
            step: Box::new(AstNode::Const(1_isize.into())),
            stop: Box::new(AstNode::Const(10_isize.into())),
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

        assert!(code.contains("for (int i = 0; i < 10; i += 1)"));
        assert!(code.contains("shared[i] = input[i]"));
        assert!(code.contains("threadgroup_barrier"));
        assert!(code.contains("output[i] = shared[i]"));
    }
}
