use crate::ast::{AstNode, DType, FunctionKind};
use crate::backend::Renderer;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::opencl::{LIBLOADING_WRAPPER_NAME, OpenCLCode};

/// OpenCLレンダラー
///
/// ASTをOpenCLカーネル + ホストコードに変換します
#[derive(Debug, Clone)]
pub struct OpenCLRenderer {
    indent_level: usize,
}

impl OpenCLRenderer {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// Programをレンダリング
    pub fn render_program(&mut self, program: &AstNode) -> OpenCLCode {
        if let AstNode::Program {
            functions,
            entry_point,
        } = program
        {
            let mut code = String::new();

            // 1. ヘッダー
            code.push_str(&self.render_header());
            code.push_str("\n\n");

            // 2. カーネル関数を文字列として生成
            let mut kernel_sources = Vec::new();
            for func in functions {
                if let AstNode::Function {
                    kind,
                    name: Some(func_name),
                    ..
                } = func
                {
                    // カーネル関数のみを処理（FunctionKind::Kernel(_)）
                    if matches!(kind, FunctionKind::Kernel(_)) {
                        // カーネル関数をOpenCL形式で生成
                        let kernel_source = self.render_kernel_function(func);
                        kernel_sources.push((func_name.clone(), kernel_source));
                    }
                }
            }

            // 3. カーネルソースを文字列リテラルとして埋め込み
            for (i, (name, source)) in kernel_sources.iter().enumerate() {
                code.push_str(&format!("// Kernel source for {}\n", name));
                code.push_str(&format!("const char* kernel_source_{} = \n", i));

                // OpenCLカーネルソースを文字列リテラルとして埋め込み
                let escaped_source = source
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n\"\n\"");
                code.push_str(&format!("\"{}\";\n\n", escaped_source));
            }

            // 4. libloading用のエントリーポイント関数を生成
            code.push_str(&self.generate_host_code(entry_point, &kernel_sources));

            OpenCLCode::new(code)
        } else {
            OpenCLCode::new(String::new())
        }
    }

    /// ヘッダーを生成
    fn render_header(&self) -> String {
        let mut header = String::new();

        // プラットフォーム依存のOpenCLヘッダー
        #[cfg(target_os = "macos")]
        {
            header.push_str("#include <OpenCL/opencl.h>\n");
        }

        #[cfg(not(target_os = "macos"))]
        {
            header.push_str("#include <CL/cl.h>\n");
        }

        header.push_str("#include <stdio.h>\n");
        header.push_str("#include <stdlib.h>\n");
        header.push_str("#include <string.h>\n");

        header
    }

    /// カーネル関数をOpenCL形式でレンダリング
    fn render_kernel_function(&mut self, func: &AstNode) -> String {
        if let AstNode::Function {
            name: Some(func_name),
            params,
            body: _,
            ..
        } = func
        {
            let mut kernel_code = String::new();

            // カーネル関数のシグネチャ
            kernel_code.push_str(&format!("__kernel void {}(", func_name));

            // パラメータ
            let param_strs: Vec<String> = params
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let type_str = self.render_opencl_param_type(&param.dtype);
                    format!("{} arg{}", type_str, i)
                })
                .collect();

            kernel_code.push_str(&param_strs.join(", "));
            kernel_code.push_str(") {\n");

            // スレッドIDの取得
            kernel_code.push_str("    int gid = get_global_id(0);\n");

            // ボディ（簡易実装：単純なループ展開）
            kernel_code.push_str("    // TODO: Implement kernel body\n");

            kernel_code.push_str("}\n");

            kernel_code
        } else {
            String::new()
        }
    }

    /// OpenCLパラメータ型をレンダリング
    fn render_opencl_param_type(&self, dtype: &DType) -> String {
        match dtype {
            DType::Ptr(inner) => {
                let base = self.render_opencl_type(inner);
                format!("__global {}", base)
            }
            _ => self.render_opencl_type(dtype),
        }
    }

    /// OpenCL型をレンダリング
    #[allow(clippy::only_used_in_recursion)]
    fn render_opencl_type(&self, dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float*".to_string(),
            DType::Int => "int*".to_string(),
            DType::Ptr(inner) => {
                let base = self.render_opencl_type(inner);
                format!("{}*", base)
            }
            _ => "void*".to_string(),
        }
    }

    /// ホストコード（OpenCL初期化 + カーネル実行）を生成
    fn generate_host_code(
        &self,
        _entry_point: &str,
        kernel_sources: &[(String, String)],
    ) -> String {
        let mut code = String::new();

        code.push_str("// === OpenCL Host Code ===\n");
        code.push_str(&format!(
            "void {}(void** buffers) {{\n",
            LIBLOADING_WRAPPER_NAME
        ));
        code.push_str("    cl_int err;\n");
        code.push_str("    \n");

        // 1. プラットフォームとデバイスの取得
        code.push_str("    // Get platform and device\n");
        code.push_str("    cl_platform_id platform;\n");
        code.push_str("    clGetPlatformIDs(1, &platform, NULL);\n");
        code.push_str("    \n");
        code.push_str("    cl_device_id device;\n");
        code.push_str("    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);\n");
        code.push_str("    \n");

        // 2. コンテキストとキューの作成
        code.push_str("    // Create context and command queue\n");
        code.push_str(
            "    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);\n",
        );
        code.push_str(
            "    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);\n",
        );
        code.push_str("    \n");

        // 3. カーネルのビルド（最初のカーネルソースを使用）
        if !kernel_sources.is_empty() {
            code.push_str("    // Build kernel\n");
            code.push_str("    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_0, NULL, &err);\n");
            code.push_str("    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);\n");
            code.push_str("    \n");
            code.push_str("    // Check for build errors\n");
            code.push_str("    if (err != CL_SUCCESS) {\n");
            code.push_str("        size_t log_size;\n");
            code.push_str("        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);\n");
            code.push_str("        char* log = (char*)malloc(log_size);\n");
            code.push_str("        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);\n");
            code.push_str("        fprintf(stderr, \"OpenCL build error:\\n%s\\n\", log);\n");
            code.push_str("        free(log);\n");
            code.push_str("    }\n");
            code.push_str("    \n");

            let kernel_name = &kernel_sources[0].0;
            code.push_str(&format!(
                "    cl_kernel kernel = clCreateKernel(program, \"{}\", &err);\n",
                kernel_name
            ));
            code.push_str("    \n");
        }

        // 4. TODO: バッファの作成とカーネル実行
        code.push_str("    // TODO: Create buffers and execute kernel\n");
        code.push_str("    // This is a simplified version - full implementation needed\n");
        code.push_str("    \n");

        // 5. クリーンアップ
        code.push_str("    // Cleanup\n");
        if !kernel_sources.is_empty() {
            code.push_str("    clReleaseKernel(kernel);\n");
            code.push_str("    clReleaseProgram(program);\n");
        }
        code.push_str("    clReleaseCommandQueue(queue);\n");
        code.push_str("    clReleaseContext(context);\n");
        code.push_str("}\n");

        code
    }
}

impl Default for OpenCLRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for OpenCLRenderer {
    type CodeRepr = OpenCLCode;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut renderer = self.clone();
        renderer.render_program(program)
    }

    fn is_available(&self) -> bool {
        // OpenCLヘッダーの存在チェック（簡易版）
        true
    }
}

impl CLikeRenderer for OpenCLRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Int => "int".to_string(),
            DType::Ptr(inner) => {
                let base = self.render_dtype_backend(inner);
                format!("__global {}*", base)
            }
            DType::Vec(inner, size) => {
                // OpenCLのベクトル型
                let base = self.render_dtype_backend(inner);
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
                panic!(
                    "Type inference failed: DType::Unknown should not appear in code generation."
                )
            }
        }
    }

    fn render_barrier_backend(&self) -> String {
        // OpenCLのワークグループバリア
        "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);".to_string()
    }

    fn render_header(&self) -> String {
        let mut header = String::new();

        // OpenCLカーネルのヘルパーマクロ
        header.push_str("// OpenCL Kernel Code\n");
        header.push_str("// Generated by Harp\n\n");

        // SIMD/ベクトル型はOpenCLで標準サポート
        header.push_str(
            "// OpenCL built-in vector types: float2, float4, float8, int2, int4, int8\n\n",
        );

        header
    }

    fn render_function_qualifier(&self, func_kind: &FunctionKind) -> String {
        match func_kind {
            FunctionKind::Kernel(_) => "__kernel ".to_string(),
            FunctionKind::Normal => String::new(),
        }
    }

    fn render_param_attribute(&self, param: &crate::ast::VarDecl, _is_kernel: bool) -> String {
        use crate::ast::{Mutability, VarKind};

        let type_str = self.render_dtype_backend(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        // ThreadId, GroupIdなどは関数内でget_global_id()等で取得するため、パラメータには含めない
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

    fn render_thread_var_declarations(
        &self,
        _params: &[crate::ast::VarDecl],
        indent: &str,
    ) -> String {
        // OpenCLではget_global_id()を使用してスレッドIDを取得
        format!(
            "{}int __thread_id = get_global_id(0);\n{}int __local_id = get_local_id(0);\n{}int __group_id = get_group_id(0);\n",
            indent, indent, indent
        )
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        // OpenCLは標準的な数学関数を提供
        match name {
            "max" => {
                if args.len() == 2 {
                    format!("fmax({}, {})", args[0], args[1])
                } else {
                    format!("max({})", args.join(", "))
                }
            }
            "min" => {
                if args.len() == 2 {
                    format!("fmin({}, {})", args[0], args[1])
                } else {
                    format!("min({})", args.join(", "))
                }
            }
            "sqrt" => format!("sqrt({})", args[0]),
            "exp2" => format!("exp2({})", args[0]),
            "log2" => format!("log2({})", args[0]),
            "sin" => format!("sin({})", args[0]),
            "cos" => format!("cos({})", args[0]),
            "rsqrt" => format!("rsqrt({})", args[0]), // OpenCLにはrsqrtがある
            _ => format!("{}({})", name, args.join(", ")),
        }
    }

    fn render_vector_load(&self, ptr_expr: &str, offset_expr: &str, dtype: &str) -> String {
        // OpenCLのvload関数を使用
        let vec_size = dtype.chars().last().and_then(|c| c.to_digit(10));
        match vec_size {
            Some(n) => format!("vload{}({} / {}, {})", n, offset_expr, n, ptr_expr),
            None => format!("{}[{}]", ptr_expr, offset_expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Mutability, Scope, VarDecl, VarKind};
    use crate::backend::c_like::CLikeRenderer;

    #[test]
    fn test_render_header() {
        let renderer = OpenCLRenderer::new();
        let header = renderer.render_header();

        #[cfg(target_os = "macos")]
        assert!(header.contains("#include <OpenCL/opencl.h>"));

        #[cfg(not(target_os = "macos"))]
        assert!(header.contains("#include <CL/cl.h>"));
    }

    #[test]
    fn test_opencl_type_rendering() {
        let renderer = OpenCLRenderer::new();

        assert_eq!(renderer.render_opencl_type(&DType::F32), "float*");
        assert_eq!(renderer.render_opencl_type(&DType::Int), "int*");
    }

    #[test]
    fn test_kernel_function_with_mixed_params() {
        // カーネル関数でThreadIdとNormalパラメータが混在している場合のテスト
        let mut renderer = OpenCLRenderer::new();

        let params = vec![
            VarDecl {
                name: "tid".to_string(),
                dtype: DType::Int,
                kind: VarKind::ThreadId(0), // フィルタされるべき
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "input".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                kind: VarKind::Normal,
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "output".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                kind: VarKind::Normal,
                mutability: Mutability::Mutable,
            },
        ];

        let scope = Scope::new();
        let body = Box::new(AstNode::Block {
            statements: vec![],
            scope: Box::new(scope),
        });

        let func = AstNode::Function {
            name: Some("test_kernel".to_string()),
            params,
            return_type: DType::Tuple(vec![]),
            body,
            kind: FunctionKind::Kernel(64),
        };

        let rendered = renderer.render_function_node(&func);

        // ThreadIdパラメータはフィルタされ、カンマが正しく処理されること
        assert!(rendered.contains("__kernel"));
        assert!(rendered.contains("test_kernel"));
        // パラメータ部分に不正なカンマが無いこと
        assert!(!rendered.contains("(,"));
        assert!(!rendered.contains(", ,"));
        // 正常なパラメータが含まれること
        assert!(rendered.contains("input"));
        assert!(rendered.contains("output"));
    }
}
