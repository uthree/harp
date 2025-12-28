use super::Renderer;
use super::c_like::CLikeRenderer;
use crate::ast::{AstNode, DType, Mutability, VarDecl, VarKind};

/// Metal Shading Language のソースコードを表す型
///
/// newtype pattern を使用して、型システムで Metal 専用のコードとして扱う。
/// これにより、誤って他のバックエンドにコードを渡すことを防ぐ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalCode(String);

impl MetalCode {
    /// 新しい MetalCode を作成
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

impl From<String> for MetalCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<MetalCode> for String {
    fn from(code: MetalCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for MetalCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for MetalCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================
// Metal共通のレンダリングヘルパー関数
// MetalRendererとMetalKernelRendererの両方で使用
// ============================================================

/// Metal用のDType変換
fn render_dtype_metal(dtype: &DType) -> String {
    match dtype {
        DType::Bool => "uchar".to_string(),
        DType::I8 => "char".to_string(),
        DType::I16 => "short".to_string(),
        DType::I32 => "int".to_string(),
        DType::I64 => "long".to_string(),
        DType::U8 => "uchar".to_string(),
        DType::U16 => "ushort".to_string(),
        DType::U32 => "uint".to_string(),
        DType::U64 => "ulong".to_string(),
        DType::F32 => "float".to_string(),
        DType::F64 => "double".to_string(), // Note: Metal has limited double support
        DType::Int => "int".to_string(),    // Index type: 32-bit for GPU efficiency
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
///
/// GroupIdとLocalIdは複数存在する可能性があるため、ここでは空文字を返す。
/// 代わりに `render_extra_kernel_params_metal` で統合された uint3 パラメータを追加し、
/// `render_thread_var_declarations_metal` で個別の変数宣言を生成する。
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
            // GroupIdとLocalIdは render_extra_kernel_params_metal で統合パラメータとして追加
            VarKind::GroupId(_) | VarKind::LocalId(_) => String::new(),
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

/// Metal用の追加カーネルパラメータ
///
/// GroupIdやLocalIdがある場合、統合されたuint3パラメータを追加する。
fn render_extra_kernel_params_metal(params: &[VarDecl]) -> Vec<String> {
    use crate::ast::VarKind;

    let mut extra_params = Vec::new();

    // GroupIdが1つ以上あれば uint3 _gid を追加
    let has_group_id = params.iter().any(|p| matches!(p.kind, VarKind::GroupId(_)));
    if has_group_id {
        extra_params.push("uint3 _gid [[threadgroup_position_in_grid]]".to_string());
    }

    // LocalIdが1つ以上あれば uint3 _lid を追加
    let has_local_id = params.iter().any(|p| matches!(p.kind, VarKind::LocalId(_)));
    if has_local_id {
        extra_params.push("uint3 _lid [[thread_position_in_threadgroup]]".to_string());
    }

    extra_params
}

/// Metal用のスレッド変数宣言
///
/// GroupIdやLocalIdパラメータに対して、uint3から個別の変数を抽出する宣言を生成する。
fn render_thread_var_declarations_metal(params: &[VarDecl], indent: &str) -> String {
    use crate::ast::VarKind;

    let mut declarations = String::new();

    for param in params {
        match &param.kind {
            VarKind::GroupId(axis) => {
                let component = match axis {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => continue,
                };
                declarations.push_str(&format!("{}uint {} = _gid.{};\n", indent, param.name, component));
            }
            VarKind::LocalId(axis) => {
                let component = match axis {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => continue,
                };
                declarations.push_str(&format!("{}uint {} = _lid.{};\n", indent, param.name, component));
            }
            _ => {}
        }
    }

    declarations
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

/// Metal Shading Language用のレンダラー
#[derive(Clone)]
pub struct MetalRenderer {
    indent_level: usize,
}

impl MetalRenderer {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// プログラム全体をMetal Shading Languageコードとして描画
    pub fn render_program(&mut self, program: &AstNode) -> MetalCode {
        if let AstNode::Program { functions, .. } = program {
            // Metal Shading Languageカーネルソースを生成
            let mut kernel_renderer = MetalKernelRenderer::new();
            let kernel_source = kernel_renderer.render_kernel_source(functions);
            MetalCode::new(kernel_source)
        } else {
            panic!("Expected AstNode::Program");
        }
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

        // カーネル関数とFunction関数をレンダリング
        for func in functions {
            match func {
                AstNode::Kernel { name: Some(_), .. } => {
                    code.push_str(&self.render_function_node(func));
                    code.push('\n');
                }
                AstNode::Function { name: Some(_), .. } => {
                    // FunctionノードをMetalカーネルとしてレンダリング
                    code.push_str(&self.render_sequential_function_as_kernel(func));
                    code.push('\n');
                }
                _ => {}
            }
        }

        code
    }

    /// 逐次関数をMetalカーネルとしてレンダリング
    ///
    /// AstNode::FunctionをMetal Shading Languageのkernel関数として変換します。
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

            // kernel修飾子を追加
            code.push_str("kernel ");

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
                use crate::renderer::c_like::extract_buffer_placeholders;
                let (inputs, has_output) = extract_buffer_placeholders(body);
                let mut buffer_params: Vec<String> = Vec::new();
                let mut buffer_index = 0;

                // 入力バッファパラメータを生成（device const float*）
                for input_name in &inputs {
                    buffer_params.push(format!(
                        "device const float* {} [[buffer({})]]",
                        input_name, buffer_index
                    ));
                    buffer_index += 1;
                }

                // 出力バッファパラメータを生成（device float*）
                if has_output {
                    buffer_params
                        .push(format!("device float* output [[buffer({})]]", buffer_index));
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

            // DEBUG: Print body structure before rendering
            fn count_stores(node: &AstNode) -> usize {
                match node {
                    AstNode::Store { .. } => 1,
                    AstNode::Block { statements, .. } => statements.iter().map(count_stores).sum(),
                    AstNode::Range { body, .. } => count_stores(body),
                    AstNode::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        count_stores(then_body) + else_body.as_ref().map_or(0, |b| count_stores(b))
                    }
                    _ => 0,
                }
            }
            fn count_statements(node: &AstNode) -> usize {
                match node {
                    AstNode::Block { statements, .. } => statements.len(),
                    _ => 1,
                }
            }
            eprintln!(
                "render_sequential_function_as_kernel: body has {} top-level statements, {} total Stores",
                count_statements(body),
                count_stores(body)
            );
            if let AstNode::Block { statements, .. } = body.as_ref() {
                for (i, stmt) in statements.iter().enumerate() {
                    eprintln!(
                        "render_sequential_function_as_kernel:   body[{}]: {:?} (Stores: {})",
                        i,
                        std::mem::discriminant(stmt),
                        count_stores(stmt)
                    );
                }
            }

            // 関数本体
            self.inc_indent();
            code.push_str(&self.render_statement(body));
            self.dec_indent();

            code.push_str("}\n");
            eprintln!("Generated Metal kernel:\n{}", code);
            code
        } else {
            String::new()
        }
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

    fn render_thread_var_declarations(&self, params: &[VarDecl], indent: &str) -> String {
        render_thread_var_declarations_metal(params, indent)
    }

    fn render_extra_kernel_params(&self, params: &[VarDecl]) -> Vec<String> {
        render_extra_kernel_params_metal(params)
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        render_math_func_metal(name, args)
    }

    fn render_vector_load(&self, ptr_expr: &str, offset_expr: &str, dtype: &str) -> String {
        // Metalではポインタキャストを使用してベクトルロード
        // *((device float2*)(ptr + offset))
        format!(
            "*((device const {}*)({} + {}))",
            dtype, ptr_expr, offset_expr
        )
    }

    fn render_vector_store(
        &self,
        ptr_expr: &str,
        offset_expr: &str,
        value_expr: &str,
        dtype: &str,
    ) -> String {
        // Metalではポインタキャストを使用してベクトルストア
        // *((device float2*)(ptr + offset)) = value
        format!(
            "*((device {}*)({} + {})) = {}",
            dtype, ptr_expr, offset_expr, value_expr
        )
    }

    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String {
        match dtype {
            DType::I64 => {
                format!(
                    "atomic_fetch_add_explicit((device atomic_int*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
            }
            DType::F32 => {
                format!(
                    "atomic_fetch_add_explicit((device atomic<float>*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
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
                format!(
                    "atomic_fetch_max_explicit((device atomic_int*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
            }
            DType::F32 => {
                format!(
                    "atomic_fetch_max_explicit((device atomic<float>*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
            }
            _ => format!(
                "/* unsupported atomic_max for {:?} */ {}[{}] = max({}[{}], {})",
                dtype, ptr, offset, ptr, offset, value
            ),
        }
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

    fn render_thread_var_declarations(&self, params: &[VarDecl], indent: &str) -> String {
        render_thread_var_declarations_metal(params, indent)
    }

    fn render_extra_kernel_params(&self, params: &[VarDecl]) -> Vec<String> {
        render_extra_kernel_params_metal(params)
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        render_math_func_metal(name, args)
    }

    fn render_vector_load(&self, ptr_expr: &str, offset_expr: &str, dtype: &str) -> String {
        // Metalではポインタキャストを使用してベクトルロード
        // *((device float2*)(ptr + offset))
        format!(
            "*((device const {}*)({} + {}))",
            dtype, ptr_expr, offset_expr
        )
    }

    fn render_vector_store(
        &self,
        ptr_expr: &str,
        offset_expr: &str,
        value_expr: &str,
        dtype: &str,
    ) -> String {
        // Metalではポインタキャストを使用してベクトルストア
        // *((device float2*)(ptr + offset)) = value
        format!(
            "*((device {}*)({} + {})) = {}",
            dtype, ptr_expr, offset_expr, value_expr
        )
    }

    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String {
        match dtype {
            DType::I64 => {
                // Metalのatomic_fetch_add_explicit
                format!(
                    "atomic_fetch_add_explicit((device atomic_int*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
            }
            DType::F32 => {
                // Metal 3.0以降でfloatのアトミック操作がサポートされている
                format!(
                    "atomic_fetch_add_explicit((device atomic<float>*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
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
                format!(
                    "atomic_fetch_max_explicit((device atomic_int*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
            }
            DType::F32 => {
                // Metal 3.0以降でfloatのアトミック操作がサポートされている
                format!(
                    "atomic_fetch_max_explicit((device atomic<float>*)&{}[{}], {}, memory_order_relaxed)",
                    ptr, offset, value
                )
            }
            _ => format!(
                "/* unsupported atomic_max for {:?} */ {}[{}] = max({}[{}], {})",
                dtype, ptr, offset, ptr, offset, value
            ),
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
impl crate::backend::pipeline::KernelSourceRenderer for MetalRenderer {
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
    use crate::renderer::c_like::CLikeRenderer;

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_render_literal() {
        let renderer = MetalRenderer::new();
        assert_eq!(renderer.render_literal(&Literal::Bool(true)), "1");
        assert_eq!(renderer.render_literal(&Literal::Bool(false)), "0");
        assert_eq!(renderer.render_literal(&Literal::F32(3.14)), "3.14f");
        assert_eq!(renderer.render_literal(&Literal::I64(42)), "42");
    }

    #[test]
    fn test_render_dtype() {
        let renderer = MetalRenderer::new();
        assert_eq!(renderer.render_dtype_backend(&DType::Bool), "uchar");
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::I64), "long");
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
                name: "gidx".to_string(),
                dtype: DType::I64,
                mutability: Mutability::Immutable,
                kind: VarKind::GroupId(0),
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
            var("gidx"),
            load(var("input"), var("gidx"), DType::F32) * AstNode::Const(2.0f32.into()),
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
        assert!(code.contains("threadgroup_position_in_grid"));
        assert!(code.contains("device float*"));
        assert!(code.contains("output[gidx] = (input[gidx] * 2.0f)"));
    }

    #[test]
    fn test_render_program() {
        use crate::ast::Scope;
        use crate::ast::helper::const_int;
        let one = const_int(1);

        // カーネル関数を作成
        let kernel_params = vec![
            VarDecl {
                name: "gidx".to_string(),
                dtype: DType::I64,
                mutability: Mutability::Immutable,
                kind: VarKind::GroupId(0),
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
            execution_waves: vec![],
        };

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_program(&program);

        // Metal Shading Languageヘッダーをチェック
        assert!(code.contains("#include <metal_stdlib>"));
        assert!(code.contains("using namespace metal;"));

        // カーネル関数が含まれていることをチェック
        assert!(code.contains("kernel"));
        assert!(code.contains("test_kernel"));
    }

    #[test]
    fn test_render_loop_with_barrier() {
        // ループとバリアを含むカーネル
        let mut loop_scope = Scope::new();
        loop_scope
            .declare("i".to_string(), DType::I64, Mutability::Immutable)
            .unwrap();

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0_i64.into())),
            step: Box::new(AstNode::Const(1_i64.into())),
            stop: Box::new(AstNode::Const(10_i64.into())),
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

        assert!(code.contains("for (long i = 0; i < 10; i += 1)"));
        assert!(code.contains("shared[i] = input[i]"));
        assert!(code.contains("threadgroup_barrier"));
        assert!(code.contains("output[i] = shared[i]"));
    }
}
