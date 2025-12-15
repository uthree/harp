use crate::ast::{AstNode, DType, Literal};
use crate::backend::Renderer;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::opencl::{LIBLOADING_WRAPPER_NAME, OpenCLCode};
use std::collections::HashMap;

/// OpenCLレンダラー
///
/// ASTをOpenCLカーネル + ホストコードに変換します
#[derive(Debug, Clone)]
pub struct OpenCLRenderer {
    indent_level: usize,
    /// シェイプ変数のデフォルト値
    shape_var_defaults: HashMap<String, isize>,
}

impl OpenCLRenderer {
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            shape_var_defaults: HashMap::new(),
        }
    }

    /// シェイプ変数のデフォルト値を設定
    pub fn with_shape_var_defaults(mut self, defaults: HashMap<String, isize>) -> Self {
        self.shape_var_defaults = defaults;
        self
    }

    /// シェイプ変数のデフォルト値を追加
    pub fn set_shape_var_default(&mut self, name: impl Into<String>, value: isize) {
        self.shape_var_defaults.insert(name.into(), value);
    }

    /// Programをレンダリング
    ///
    /// カーネル関数群をOpenCLコードとして出力します。
    /// カーネルの実行順序はホスト側（CompiledProgram）で管理されます。
    pub fn render_program(&mut self, program: &AstNode) -> OpenCLCode {
        if let AstNode::Program { functions } = program {
            let mut code = String::new();

            // 1. ヘッダー
            code.push_str(&self.render_header());
            code.push_str("\n\n");

            // 2. カーネル関数を文字列として生成
            // AstNode::Kernel (並列) と AstNode::Function (逐次) の両方をサポート
            let mut kernel_sources = Vec::new();

            for func in functions {
                match func {
                    AstNode::Kernel {
                        name: Some(func_name),
                        ..
                    } => {
                        // 並列カーネル関数をOpenCL形式で生成
                        let kernel_source = self.render_kernel_function(func);
                        kernel_sources.push((func_name.clone(), kernel_source, func));
                    }
                    AstNode::Kernel { name: None, .. } => {
                        log::warn!("OpenCLRenderer: Kernel with no name found");
                    }
                    AstNode::Function {
                        name: Some(func_name),
                        ..
                    } => {
                        // 逐次関数をOpenCLカーネルとして生成
                        // OpenCLでは逐次関数も__kernelとして実行される
                        let kernel_source = self.render_sequential_function_as_kernel(func);
                        kernel_sources.push((func_name.clone(), kernel_source, func));
                    }
                    AstNode::Function { name: None, .. } => {
                        log::warn!("OpenCLRenderer: Function with no name found");
                    }
                    _ => {}
                }
            }

            // 3. カーネルソースを文字列リテラルとして埋め込み
            for (i, (name, source, _)) in kernel_sources.iter().enumerate() {
                code.push_str(&format!("// Kernel source for {}\n", name));
                code.push_str(&format!("const char* kernel_source_{} = \n", i));

                // OpenCLカーネルソースを文字列リテラルとして埋め込み
                let escaped_source = source
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n\"\n\"");
                code.push_str(&format!("\"{}\";\n\n", escaped_source));
            }

            // 4. libloading用のホストコード生成（最初のカーネルをデフォルトとして使用）
            let first_kernel = kernel_sources.first().map(|(_, _, func)| *func);
            let kernel_name_sources: Vec<(String, String)> = kernel_sources
                .iter()
                .map(|(name, source, _)| (name.clone(), source.clone()))
                .collect();
            code.push_str(&self.generate_host_code(first_kernel, &kernel_name_sources));

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
    ///
    /// CLikeRendererの共通実装を使用して、関数本体を正しくレンダリングします。
    fn render_kernel_function(&mut self, func: &AstNode) -> String {
        // CLikeRendererの共通実装を使用
        self.render_function_node(func)
    }

    /// 逐次関数をOpenCLカーネルとしてレンダリング
    ///
    /// AstNode::FunctionをOpenCLの__kernel関数として変換します。
    /// 関数本体はそのままレンダリングされ、単一ワークアイテムで実行されます。
    fn render_sequential_function_as_kernel(&mut self, func: &AstNode) -> String {
        if let AstNode::Function {
            name,
            params,
            return_type,
            body,
        } = func
        {
            let mut code = String::new();

            // __kernel修飾子を追加
            code.push_str("__kernel ");

            // 戻り値型
            code.push_str(&self.render_dtype_backend(return_type));
            code.push(' ');

            // 関数名
            if let Some(n) = name {
                code.push_str(n);
            }

            // パラメータ
            code.push('(');
            let rendered_params: Vec<String> = params
                .iter()
                .map(|p| self.render_param_attribute(p, true))
                .filter(|s| !s.is_empty())
                .collect();
            code.push_str(&rendered_params.join(", "));
            code.push_str(") {\n");

            // 関数本体をレンダリング
            *self.indent_level_mut() += 1;
            code.push_str(&self.render_statement(body));
            *self.indent_level_mut() -= 1;

            code.push_str("}\n");

            code
        } else {
            String::new()
        }
    }

    /// ホストコード（OpenCL初期化 + カーネル実行）を生成
    ///
    /// バッファ情報を元に完全なOpenCLホストコードを生成します。
    /// シグネチャ: void __harp_entry(void** buffers, size_t* sizes, int* is_outputs, int num_buffers)
    fn generate_host_code(
        &self,
        entry_func: Option<&AstNode>,
        kernel_sources: &[(String, String)],
    ) -> String {
        let mut code = String::new();

        // エントリー関数からパラメータ情報を取得
        let (params, grid_size, thread_group_size) = if let Some(func) = entry_func {
            match func {
                AstNode::Kernel {
                    params,
                    default_grid_size,
                    default_thread_group_size,
                    ..
                } => (
                    params.clone(),
                    default_grid_size.clone(),
                    default_thread_group_size.clone(),
                ),
                AstNode::Function { params, .. } => {
                    use crate::ast::helper::const_int;
                    let one = const_int(1);
                    (
                        params.clone(),
                        [Box::new(one.clone()), Box::new(one.clone()), Box::new(one)],
                        [
                            Box::new(const_int(1)),
                            Box::new(const_int(1)),
                            Box::new(const_int(1)),
                        ],
                    )
                }
                _ => return self.generate_fallback_host_code(kernel_sources),
            }
        } else {
            return self.generate_fallback_host_code(kernel_sources);
        };

        // バッファパラメータのみをフィルタリング（Ptr型のパラメータ）
        let buffer_params: Vec<_> = params
            .iter()
            .filter(|p| matches!(p.dtype, DType::Ptr(_)))
            .collect();

        code.push_str("// === OpenCL Host Code ===\n");
        code.push_str(&format!(
            "void {}(void** buffers, size_t* sizes, int* is_outputs, int num_buffers) {{\n",
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

        // 3. カーネルのビルド
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
            code.push_str("        clReleaseCommandQueue(queue);\n");
            code.push_str("        clReleaseContext(context);\n");
            code.push_str("        return;\n");
            code.push_str("    }\n");
            code.push_str("    \n");

            let kernel_name = &kernel_sources[0].0;
            code.push_str(&format!(
                "    cl_kernel kernel = clCreateKernel(program, \"{}\", &err);\n",
                kernel_name
            ));
            code.push_str("    \n");

            // 4. OpenCLバッファの作成
            code.push_str("    // Create OpenCL buffers\n");
            code.push_str(&format!(
                "    cl_mem cl_buffers[{}];\n",
                buffer_params.len().max(1)
            ));
            code.push_str("    for (int i = 0; i < num_buffers; i++) {\n");
            code.push_str("        cl_mem_flags flags = is_outputs[i] ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY;\n");
            code.push_str("        cl_buffers[i] = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, sizes[i], buffers[i], &err);\n");
            code.push_str("        if (err != CL_SUCCESS) {\n");
            code.push_str(
                "            fprintf(stderr, \"Failed to create buffer %d: %d\\n\", i, err);\n",
            );
            code.push_str("        }\n");
            code.push_str("    }\n");
            code.push_str("    \n");

            // 5. カーネル引数の設定
            code.push_str("    // Set kernel arguments\n");
            code.push_str("    for (int i = 0; i < num_buffers; i++) {\n");
            code.push_str(
                "        err = clSetKernelArg(kernel, i, sizeof(cl_mem), &cl_buffers[i]);\n",
            );
            code.push_str("        if (err != CL_SUCCESS) {\n");
            code.push_str(
                "            fprintf(stderr, \"Failed to set kernel arg %d: %d\\n\", i, err);\n",
            );
            code.push_str("        }\n");
            code.push_str("    }\n");
            code.push_str("    \n");

            // 6. カーネル実行（グリッドサイズ計算）
            code.push_str("    // Execute kernel\n");

            // グリッドサイズとスレッドグループサイズを評価
            let grid_x = self.eval_const_expr(&grid_size[0]);
            let grid_y = self.eval_const_expr(&grid_size[1]);
            let grid_z = self.eval_const_expr(&grid_size[2]);
            let group_x = self.eval_const_expr(&thread_group_size[0]);
            let group_y = self.eval_const_expr(&thread_group_size[1]);
            let group_z = self.eval_const_expr(&thread_group_size[2]);

            // global_work_size = grid_size * thread_group_size
            code.push_str(&format!(
                "    size_t global_work_size[3] = {{{}, {}, {}}};\n",
                grid_x * group_x,
                grid_y * group_y,
                grid_z * group_z
            ));
            code.push_str(&format!(
                "    size_t local_work_size[3] = {{{}, {}, {}}};\n",
                group_x, group_y, group_z
            ));

            // 次元数を計算（1以外の値がある次元のみ使用）
            let work_dim = if grid_z * group_z > 1 {
                3
            } else if grid_y * group_y > 1 {
                2
            } else {
                1
            };

            code.push_str(&format!(
                "    err = clEnqueueNDRangeKernel(queue, kernel, {}, NULL, global_work_size, local_work_size, 0, NULL, NULL);\n",
                work_dim
            ));
            code.push_str("    if (err != CL_SUCCESS) {\n");
            code.push_str("        fprintf(stderr, \"Failed to execute kernel: %d\\n\", err);\n");
            code.push_str("    }\n");
            code.push_str("    \n");

            // 7. 結果の読み出し（出力バッファのみ）
            code.push_str("    // Read back output buffers\n");
            code.push_str("    for (int i = 0; i < num_buffers; i++) {\n");
            code.push_str("        if (is_outputs[i]) {\n");
            code.push_str("            err = clEnqueueReadBuffer(queue, cl_buffers[i], CL_TRUE, 0, sizes[i], buffers[i], 0, NULL, NULL);\n");
            code.push_str("            if (err != CL_SUCCESS) {\n");
            code.push_str(
                "                fprintf(stderr, \"Failed to read buffer %d: %d\\n\", i, err);\n",
            );
            code.push_str("            }\n");
            code.push_str("        }\n");
            code.push_str("    }\n");
            code.push_str("    \n");

            // 8. OpenCLバッファの解放
            code.push_str("    // Release OpenCL buffers\n");
            code.push_str("    for (int i = 0; i < num_buffers; i++) {\n");
            code.push_str("        clReleaseMemObject(cl_buffers[i]);\n");
            code.push_str("    }\n");
            code.push_str("    \n");

            // 9. クリーンアップ
            code.push_str("    // Cleanup\n");
            code.push_str("    clReleaseKernel(kernel);\n");
            code.push_str("    clReleaseProgram(program);\n");
        }

        code.push_str("    clReleaseCommandQueue(queue);\n");
        code.push_str("    clReleaseContext(context);\n");
        code.push_str("}\n");

        code
    }

    /// フォールバック用のホストコード生成（エントリー関数が見つからない場合）
    fn generate_fallback_host_code(&self, kernel_sources: &[(String, String)]) -> String {
        let mut code = String::new();

        code.push_str("// === OpenCL Host Code (Fallback) ===\n");
        code.push_str(&format!(
            "void {}(void** buffers, size_t* sizes, int* is_outputs, int num_buffers) {{\n",
            LIBLOADING_WRAPPER_NAME
        ));
        code.push_str("    (void)buffers; (void)sizes; (void)is_outputs; (void)num_buffers;\n");
        code.push_str(
            "    fprintf(stderr, \"Warning: Fallback host code - kernel execution not implemented\\n\");\n",
        );

        if !kernel_sources.is_empty() {
            code.push_str(&format!(
                "    // Kernel source available: {}\\n\",\n",
                kernel_sources[0].0
            ));
        }

        code.push_str("}\n");

        code
    }

    /// 定数式を評価してusize値を取得
    ///
    /// 変数（Var）が含まれている場合は`shape_var_defaults`の値で置換してから評価します。
    /// 評価できない場合は1を返します。
    fn eval_const_expr(&self, expr: &AstNode) -> usize {
        // まず変数を定数に置換
        let substituted = if !self.shape_var_defaults.is_empty() {
            let mappings: HashMap<String, AstNode> = self
                .shape_var_defaults
                .iter()
                .map(|(name, value)| (name.clone(), AstNode::Const(Literal::Int(*value))))
                .collect();
            expr.substitute_vars(&mappings)
        } else {
            expr.clone()
        };

        // 簡略化された式を評価
        self.eval_simplified_expr(&substituted).unwrap_or(1)
    }

    /// 簡略化された（変数のない）式を評価
    fn eval_simplified_expr(&self, expr: &AstNode) -> Option<usize> {
        match expr {
            AstNode::Const(Literal::Int(n)) => {
                if *n >= 0 {
                    Some(*n as usize)
                } else {
                    None
                }
            }
            AstNode::Add(left, right) => {
                let l = self.eval_simplified_expr(left)?;
                let r = self.eval_simplified_expr(right)?;
                Some(l + r)
            }
            AstNode::Mul(left, right) => {
                let l = self.eval_simplified_expr(left)?;
                let r = self.eval_simplified_expr(right)?;
                Some(l * r)
            }
            AstNode::Idiv(left, right) => {
                let l = self.eval_simplified_expr(left)?;
                let r = self.eval_simplified_expr(right)?;
                if r == 0 { None } else { Some(l / r) }
            }
            AstNode::Rem(left, right) => {
                let l = self.eval_simplified_expr(left)?;
                let r = self.eval_simplified_expr(right)?;
                if r == 0 { None } else { Some(l % r) }
            }
            // Varが残っている場合はshape_var_defaultsから直接取得を試みる
            AstNode::Var(name) => self.shape_var_defaults.get(name).map(|v| *v as usize),
            _ => None,
        }
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
            DType::Bool => "uchar".to_string(), // OpenCLではucharを使用
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

    fn render_function_qualifier(&self, is_kernel: bool) -> String {
        if is_kernel {
            "__kernel ".to_string()
        } else {
            String::new()
        }
    }

    fn render_param_attribute(&self, param: &crate::ast::VarDecl, _is_kernel: bool) -> String {
        use crate::ast::{Mutability, VarKind};

        let type_str = self.render_dtype_backend(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        // ThreadId, GroupId, LocalIdなどは関数内でget_global_id()等で取得するため、パラメータには含めない
        match &param.kind {
            VarKind::Normal => {
                format!("{}{} {}", mut_str, type_str, param.name)
            }
            VarKind::ThreadId(_)
            | VarKind::GroupId(_)
            | VarKind::LocalId(_)
            | VarKind::GroupSize(_)
            | VarKind::GridSize(_) => {
                // これらは関数内で宣言するため、ここでは空文字を返す
                String::new()
            }
        }
    }

    fn render_thread_var_declarations(
        &self,
        params: &[crate::ast::VarDecl],
        indent: &str,
    ) -> String {
        use crate::ast::VarKind;

        let mut declarations = String::new();

        // パラメータからThreadId/GroupId/LocalId等を探し、実際の名前を使用する
        for param in params {
            match &param.kind {
                VarKind::ThreadId(axis) => {
                    declarations.push_str(&format!(
                        "{}int {} = get_global_id({});\n",
                        indent, param.name, axis
                    ));
                }
                VarKind::GroupId(axis) => {
                    declarations.push_str(&format!(
                        "{}int {} = get_group_id({});\n",
                        indent, param.name, axis
                    ));
                }
                VarKind::LocalId(axis) => {
                    declarations.push_str(&format!(
                        "{}int {} = get_local_id({});\n",
                        indent, param.name, axis
                    ));
                }
                VarKind::GroupSize(axis) => {
                    declarations.push_str(&format!(
                        "{}int {} = get_local_size({});\n",
                        indent, param.name, axis
                    ));
                }
                VarKind::GridSize(axis) => {
                    declarations.push_str(&format!(
                        "{}int {} = get_global_size({});\n",
                        indent, param.name, axis
                    ));
                }
                VarKind::Normal => {}
            }
        }

        declarations
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

    fn libloading_wrapper_name(&self) -> &'static str {
        LIBLOADING_WRAPPER_NAME
    }

    fn render_libloading_wrapper(&self, _entry_func: &AstNode, _entry_point: &str) -> String {
        // OpenCLRendererは独自のrender_programを使用し、render_program_clikeを使用しないため、
        // このメソッドは直接呼ばれない。generate_host_codeで同等の処理を行っている。
        // トレイト要件を満たすためのスタブ実装。
        String::new()
    }
}

/// Implementation of KernelSourceRenderer for native OpenCL backend
#[cfg(feature = "native-opencl")]
impl crate::backend::execution::KernelSourceRenderer for OpenCLRenderer {
    fn render_kernel_source(&mut self, program: &AstNode) -> String {
        if let AstNode::Program { functions, .. } = program {
            let mut code = String::new();

            // OpenCL kernel header (minimal, no C host includes)
            code.push_str("// OpenCL Kernel Code\n");
            code.push_str("// Generated by Harp (native backend)\n\n");

            // Render all kernel/function nodes
            for func in functions {
                match func {
                    AstNode::Kernel { name: Some(_), .. } => {
                        code.push_str(&self.render_kernel_function(func));
                        code.push_str("\n\n");
                    }
                    AstNode::Function { name: Some(_), .. } => {
                        code.push_str(&self.render_sequential_function_as_kernel(func));
                        code.push_str("\n\n");
                    }
                    _ => {}
                }
            }

            code
        } else {
            String::new()
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
    fn test_kernel_function_with_mixed_params() {
        use crate::ast::helper::const_int;
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

        let one = const_int(1);
        let func = AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params,
            return_type: DType::Tuple(vec![]),
            body,
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

    #[test]
    fn test_kernel_function_all_params_filtered() {
        use crate::ast::helper::const_int;
        // 全パラメータがフィルタされる場合（ThreadId, GroupIdのみ）
        let mut renderer = OpenCLRenderer::new();

        let params = vec![
            VarDecl {
                name: "tid".to_string(),
                dtype: DType::Int,
                kind: VarKind::ThreadId(0),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "gid".to_string(),
                dtype: DType::Int,
                kind: VarKind::GroupId(0),
                mutability: Mutability::Immutable,
            },
        ];

        let scope = Scope::new();
        let body = Box::new(AstNode::Block {
            statements: vec![],
            scope: Box::new(scope),
        });

        let one = const_int(1);
        let func = AstNode::Kernel {
            name: Some("empty_params_kernel".to_string()),
            params,
            return_type: DType::Tuple(vec![]),
            body,
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

        let rendered = renderer.render_function_node(&func);

        // 空のパラメータリストになること
        assert!(rendered.contains("empty_params_kernel()"));
        assert!(!rendered.contains("(,"));
    }

    #[test]
    fn test_render_dtype_backend_variants() {
        let renderer = OpenCLRenderer::new();

        // 基本型
        assert_eq!(renderer.render_dtype_backend(&DType::Bool), "uchar");
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::Int), "int");

        // ポインタ型
        assert_eq!(
            renderer.render_dtype_backend(&DType::Ptr(Box::new(DType::F32))),
            "__global float*"
        );

        // ベクトル型
        assert_eq!(
            renderer.render_dtype_backend(&DType::Vec(Box::new(DType::F32), 4)),
            "float4"
        );
        assert_eq!(
            renderer.render_dtype_backend(&DType::Vec(Box::new(DType::Int), 2)),
            "int2"
        );

        // タプル型（空 = void）
        assert_eq!(renderer.render_dtype_backend(&DType::Tuple(vec![])), "void");
    }

    #[test]
    fn test_render_math_functions() {
        let renderer = OpenCLRenderer::new();

        // max/min
        assert_eq!(
            renderer.render_math_func("max", &["a".to_string(), "b".to_string()]),
            "fmax(a, b)"
        );
        assert_eq!(
            renderer.render_math_func("min", &["a".to_string(), "b".to_string()]),
            "fmin(a, b)"
        );

        // 単項関数
        assert_eq!(
            renderer.render_math_func("sqrt", &["x".to_string()]),
            "sqrt(x)"
        );
        assert_eq!(
            renderer.render_math_func("exp2", &["x".to_string()]),
            "exp2(x)"
        );
        assert_eq!(
            renderer.render_math_func("log2", &["x".to_string()]),
            "log2(x)"
        );
        assert_eq!(
            renderer.render_math_func("sin", &["x".to_string()]),
            "sin(x)"
        );
        assert_eq!(
            renderer.render_math_func("cos", &["x".to_string()]),
            "cos(x)"
        );
        assert_eq!(
            renderer.render_math_func("rsqrt", &["x".to_string()]),
            "rsqrt(x)"
        );
    }

    #[test]
    fn test_render_barrier() {
        let renderer = OpenCLRenderer::new();
        let barrier = renderer.render_barrier_backend();
        assert!(barrier.contains("barrier"));
        assert!(barrier.contains("CLK_LOCAL_MEM_FENCE"));
        assert!(barrier.contains("CLK_GLOBAL_MEM_FENCE"));
    }

    #[test]
    fn test_render_function_qualifier() {
        let renderer = OpenCLRenderer::new();

        assert_eq!(renderer.render_function_qualifier(true), "__kernel ");
        assert_eq!(renderer.render_function_qualifier(false), "");
    }

    #[test]
    fn test_render_vector_load() {
        let renderer = OpenCLRenderer::new();

        // ベクトル4のロード
        let load4 = renderer.render_vector_load("ptr", "offset", "float4");
        assert!(load4.contains("vload4"));

        // ベクトル2のロード
        let load2 = renderer.render_vector_load("ptr", "offset", "float2");
        assert!(load2.contains("vload2"));

        // スカラのロード
        let load_scalar = renderer.render_vector_load("ptr", "offset", "float");
        assert!(load_scalar.contains("ptr[offset]"));
    }

    #[test]
    fn test_render_thread_var_declarations() {
        use crate::ast::{DType, Mutability, VarDecl, VarKind};

        let renderer = OpenCLRenderer::new();

        // 空のパラメータリストでは何も生成されない
        let decls = renderer.render_thread_var_declarations(&[], "    ");
        assert!(decls.is_empty());

        // ThreadIdパラメータがある場合、その名前でget_global_id()を生成
        let params = vec![VarDecl {
            name: "tid".to_string(),
            dtype: DType::Int,
            kind: VarKind::ThreadId(0),
            mutability: Mutability::Immutable,
        }];
        let decls = renderer.render_thread_var_declarations(&params, "    ");
        assert!(decls.contains("int tid = get_global_id(0)"));

        // 複数のパラメータタイプ
        let params = vec![
            VarDecl {
                name: "thread_x".to_string(),
                dtype: DType::Int,
                kind: VarKind::ThreadId(0),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "thread_y".to_string(),
                dtype: DType::Int,
                kind: VarKind::ThreadId(1),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "group_x".to_string(),
                dtype: DType::Int,
                kind: VarKind::GroupId(0),
                mutability: Mutability::Immutable,
            },
        ];
        let decls = renderer.render_thread_var_declarations(&params, "    ");
        assert!(decls.contains("int thread_x = get_global_id(0)"));
        assert!(decls.contains("int thread_y = get_global_id(1)"));
        assert!(decls.contains("int group_x = get_group_id(0)"));
    }
}
