use eclat::ast::{AstNode, DType};
use eclat::backend::Renderer;
use eclat::backend::renderer::CLikeRenderer;

/// OpenCL Cコードを表す型
///
/// newtype pattern を使用して、型システムで OpenCL 専用のコードとして扱う。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCLCode(String);

impl OpenCLCode {
    /// 新しい OpenCLCode を作成
    pub fn new(code: String) -> Self {
        Self(code)
    }

    /// 内部の String への参照を取得
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// 内部の String を取得（所有権を移動）
    pub fn into_inner(self) -> String {
        self.0
    }

    /// コードのバイト数を取得
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// コードが空かどうか
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// 指定した文字列が含まれているかチェック
    pub fn contains(&self, pat: &str) -> bool {
        self.0.contains(pat)
    }
}

impl From<String> for OpenCLCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<OpenCLCode> for String {
    fn from(code: OpenCLCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for OpenCLCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for OpenCLCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// OpenCLレンダラー
///
/// ASTをOpenCLカーネルソースに変換します
#[derive(Debug, Clone)]
pub struct OpenCLRenderer {
    indent_level: usize,
}

impl OpenCLRenderer {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// Programをレンダリング
    ///
    /// カーネル関数群をOpenCLコードとして出力します。
    /// カーネルの実行順序はホスト側（CompiledProgram）で管理されます。
    pub fn render_program(&mut self, program: &AstNode) -> OpenCLCode {
        if let AstNode::Program { functions, .. } = program {
            let mut code = String::new();

            // ヘッダー
            code.push_str(&CLikeRenderer::render_header(self));
            code.push('\n');

            // カーネル関数をレンダリング
            for func in functions {
                match func {
                    AstNode::Kernel { name: Some(_), .. } => {
                        code.push_str(&self.render_kernel_function(func));
                        code.push_str("\n\n");
                    }
                    AstNode::Kernel { name: None, .. } => {
                        log::warn!("OpenCLRenderer: Kernel with no name found");
                    }
                    AstNode::Function { name: Some(_), .. } => {
                        code.push_str(&self.render_sequential_function_as_kernel(func));
                        code.push_str("\n\n");
                    }
                    AstNode::Function { name: None, .. } => {
                        log::warn!("OpenCLRenderer: Function with no name found");
                    }
                    _ => {}
                }
            }

            OpenCLCode::new(code)
        } else {
            OpenCLCode::new(String::new())
        }
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
    /// paramsが空の場合、関数本体からバッファプレースホルダーを抽出して
    /// 自動的にバッファパラメータを生成します。
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

            // paramsが空の場合、関数本体からバッファプレースホルダーを抽出
            if params.is_empty() {
                use eclat::backend::renderer::extract_buffer_placeholders;
                let (inputs, has_output) = extract_buffer_placeholders(body);
                let mut buffer_params: Vec<String> = Vec::new();

                // 入力バッファパラメータを生成（__global const float*）
                for input_name in &inputs {
                    buffer_params.push(format!("__global const float* {}", input_name));
                }

                // 出力バッファパラメータを生成（__global float*）
                if has_output {
                    buffer_params.push("__global float* output".to_string());
                }

                code.push_str(&buffer_params.join(", "));
            } else {
                let rendered_params: Vec<String> = params
                    .iter()
                    .map(|p| self.render_param_attribute(p, true))
                    .filter(|s| !s.is_empty())
                    .collect();
                code.push_str(&rendered_params.join(", "));
            }
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
            DType::Void => "void".to_string(),
            DType::Bool => "uchar".to_string(),
            DType::I8 => "char".to_string(),
            DType::I16 => "short".to_string(),
            DType::I32 => "int".to_string(),
            DType::I64 => "long".to_string(),
            DType::U8 => "uchar".to_string(),
            DType::U16 => "ushort".to_string(),
            DType::U32 => "uint".to_string(),
            DType::U64 => "ulong".to_string(),
            DType::F16 => "half".to_string(),
            DType::BF16 => "ushort".to_string(), // BF16 not natively supported, use ushort
            DType::F32 => "float".to_string(),
            DType::F64 => "double".to_string(),
            DType::Int => "int".to_string(), // Index type: 32-bit for GPU efficiency
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
            DType::Complex32 => {
                // OpenCL does not have native complex type, use float2 as (real, imag) pair
                "float2".to_string()
            }
            DType::Complex64 => {
                // OpenCL does not have native complex type, use double2 as (real, imag) pair
                "double2".to_string()
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

    fn render_param_attribute(&self, param: &eclat::ast::VarDecl, _is_kernel: bool) -> String {
        use eclat::ast::{Mutability, VarKind};

        let type_str = self.render_dtype_backend(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        // GroupId, LocalIdなどは関数内でget_group_id()等で取得するため、パラメータには含めない
        match &param.kind {
            VarKind::Normal => {
                format!("{}{} {}", mut_str, type_str, param.name)
            }
            VarKind::GroupId(_)
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
        params: &[eclat::ast::VarDecl],
        indent: &str,
    ) -> String {
        use eclat::ast::VarKind;

        let mut declarations = String::new();

        // パラメータからGroupId/LocalId等を探し、実際の名前を使用する
        for param in params {
            match &param.kind {
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

    fn render_vector_store(
        &self,
        ptr_expr: &str,
        offset_expr: &str,
        value_expr: &str,
        dtype: &str,
    ) -> String {
        // OpenCLのvstore関数を使用
        let vec_size = dtype.chars().last().and_then(|c| c.to_digit(10));
        match vec_size {
            Some(n) => format!(
                "vstore{}({}, {} / {}, {})",
                n, value_expr, offset_expr, n, ptr_expr
            ),
            None => format!("{}[{}] = {}", ptr_expr, offset_expr, value_expr),
        }
    }

    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String {
        match dtype {
            DType::I64 => {
                // OpenCLの標準atomic_add（整数用）
                format!("atomic_add(&{}[{}], {})", ptr, offset, value)
            }
            DType::F32 => {
                // OpenCLはfloatのatomic_addを標準でサポートしない
                // cl_khr_global_int32_extended_atomics拡張が必要
                // ここではCASループを使用したヘルパー関数を想定
                format!("atomic_add_float(&{}[{}], {})", ptr, offset, value)
            }
            _ => format!(
                "/* unsupported atomic_add for {:?} */ {}[{}] += {}",
                dtype, ptr, offset, value
            ),
        }
    }

    fn render_atomic_max(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String {
        match dtype {
            DType::I64 => {
                // OpenCLの標準atomic_max（整数用）
                format!("atomic_max(&{}[{}], {})", ptr, offset, value)
            }
            DType::F32 => {
                // floatのatomic_maxはCASループが必要
                format!("atomic_max_float(&{}[{}], {})", ptr, offset, value)
            }
            _ => format!(
                "/* unsupported atomic_max for {:?} */ {}[{}] = max({}[{}], {})",
                dtype, ptr, offset, ptr, offset, value
            ),
        }
    }
}

/// Implementation of KernelSourceRenderer for native OpenCL backend
impl eclat::backend::pipeline::KernelSourceRenderer for OpenCLRenderer {
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
    use eclat::ast::{Mutability, Scope, VarDecl, VarKind};
    use eclat::backend::renderer::CLikeRenderer;

    #[test]
    fn test_render_header() {
        let renderer = OpenCLRenderer::new();
        let header = CLikeRenderer::render_header(&renderer);

        // OpenCLカーネルヘッダーをチェック
        assert!(header.contains("OpenCL Kernel Code"));
        assert!(header.contains("Generated by Harp"));
    }

    #[test]
    fn test_kernel_function_with_mixed_params() {
        use eclat::ast::helper::const_int;
        // カーネル関数でGroupIdとNormalパラメータが混在している場合のテスト
        let mut renderer = OpenCLRenderer::new();

        let params = vec![
            VarDecl {
                name: "gidx".to_string(),
                dtype: DType::I64,
                kind: VarKind::GroupId(0), // フィルタされるべき
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

        // GroupIdパラメータはフィルタされ、カンマが正しく処理されること
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
        use eclat::ast::helper::const_int;
        // 全パラメータがフィルタされる場合（GroupId, LocalIdのみ）
        let mut renderer = OpenCLRenderer::new();

        let params = vec![
            VarDecl {
                name: "gidx".to_string(),
                dtype: DType::I64,
                kind: VarKind::GroupId(0),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "lid".to_string(),
                dtype: DType::I64,
                kind: VarKind::LocalId(0),
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
        assert_eq!(renderer.render_dtype_backend(&DType::I64), "long");

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
            renderer.render_dtype_backend(&DType::Vec(Box::new(DType::I64), 2)),
            "long2"
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
        use eclat::ast::{DType, Mutability, VarDecl, VarKind};

        let renderer = OpenCLRenderer::new();

        // 空のパラメータリストでは何も生成されない
        let decls = renderer.render_thread_var_declarations(&[], "    ");
        assert!(decls.is_empty());

        // GroupIdパラメータがある場合、その名前でget_group_id()を生成
        let params = vec![VarDecl {
            name: "gidx".to_string(),
            dtype: DType::I64,
            kind: VarKind::GroupId(0),
            mutability: Mutability::Immutable,
        }];
        let decls = renderer.render_thread_var_declarations(&params, "    ");
        assert!(decls.contains("int gidx = get_group_id(0)"));

        // 複数のパラメータタイプ
        let params = vec![
            VarDecl {
                name: "group_x".to_string(),
                dtype: DType::I64,
                kind: VarKind::GroupId(0),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "group_y".to_string(),
                dtype: DType::I64,
                kind: VarKind::GroupId(1),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "local_x".to_string(),
                dtype: DType::I64,
                kind: VarKind::LocalId(0),
                mutability: Mutability::Immutable,
            },
        ];
        let decls = renderer.render_thread_var_declarations(&params, "    ");
        assert!(decls.contains("int group_x = get_group_id(0)"));
        assert!(decls.contains("int group_y = get_group_id(1)"));
        assert!(decls.contains("int local_x = get_local_id(0)"));
    }
}
