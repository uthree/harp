//! Renderer module
//!
//! This module provides source code rendering for various backends.
//! Renderers are always available without feature flags, allowing
//! source code generation without GPU execution capabilities.
//!
//! ## Available Renderers
//!
//! - `GenericRenderer`: C-like generic renderer (always available)
//!
//! Backend-specific renderers (OpenCL, Metal, C) are provided by their respective
//! backend crates: `eclat-backend-opencl`, `eclat-backend-metal`, `eclat-backend-c`.

use crate::ast::{AstNode, DType, Literal, VarDecl};

/// Renderer trait for converting AST to source code
///
/// This trait is implemented by all renderers and provides a common
/// interface for source code generation.
pub trait Renderer {
    /// The type representing rendered code
    type CodeRepr: Into<String> + AsRef<str>;

    /// Renderer-specific options
    type Option;

    /// Render an AST program to source code
    fn render(&self, program: &AstNode) -> Self::CodeRepr;

    /// Check if this renderer is available on the current platform
    fn is_available(&self) -> bool;

    /// Configure renderer with an option
    fn with_option(&mut self, _option: Self::Option) {}
}

/// コンパイラ最適化レベル
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// 最適化なし（デバッグ向け、コンパイル高速）
    O0,
    /// 基本的な最適化
    O1,
    /// 標準的な最適化
    O2,
    /// アグレッシブな最適化
    #[default]
    O3,
    /// サイズ最適化
    Os,
}

impl OptimizationLevel {
    /// コンパイラフラグとして使用する文字列を返す
    pub fn as_flag(&self) -> &'static str {
        match self {
            OptimizationLevel::O0 => "0",
            OptimizationLevel::O1 => "1",
            OptimizationLevel::O2 => "2",
            OptimizationLevel::O3 => "3",
            OptimizationLevel::Os => "s",
        }
    }
}

// C言語に近い構文の言語のためのレンダラー
// Metal, CUDA, OpenCLなどのバックエンドは大体C言語に近い文法を採用しているので、共通化したい。

pub trait CLikeRenderer {
    // ========== インデント管理（実装側で提供） ==========
    fn indent_level(&self) -> usize;
    fn indent_level_mut(&mut self) -> &mut usize;
    fn indent_size(&self) -> usize {
        4 // デフォルト値
    }

    // ========== バックエンド固有のメソッド（実装側で提供） ==========

    /// 型をバックエンド固有の文字列に変換
    fn render_dtype_backend(&self, dtype: &DType) -> String;

    /// バリア命令をレンダリング
    fn render_barrier_backend(&self) -> String;

    /// プログラムのヘッダー（includeなど）をレンダリング
    fn render_header(&self) -> String;

    /// 関数修飾子（kernelなど）をレンダリング
    fn render_function_qualifier(&self, is_kernel: bool) -> String;

    /// 関数パラメータの属性（Metal用のthread_position_in_gridなど）をレンダリング
    fn render_param_attribute(&self, param: &VarDecl, is_kernel: bool) -> String;

    /// スレッドID等の特殊変数の宣言をレンダリング（OpenMP用のomp_get_thread_num()など）
    fn render_thread_var_declarations(&self, params: &[VarDecl], indent: &str) -> String;

    /// カーネル関数に追加するパラメータ（Metal用のuint3 _gidなど）
    /// GroupIdやLocalIdを含む場合に、それらを統合したパラメータを追加する
    fn render_extra_kernel_params(&self, _params: &[VarDecl]) -> Vec<String> {
        Vec::new()
    }

    /// 数学関数をレンダリング（max vs fmaxf など）
    fn render_math_func(&self, name: &str, args: &[String]) -> String;

    /// Fused Multiply-Addをレンダリング
    fn render_fma(&self, a: &str, b: &str, c: &str) -> String {
        format!("fma({}, {}, {})", a, b, c)
    }

    /// アトミック加算をレンダリング（バックエンド固有の実装が必要）
    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String;

    /// アトミック最大値をレンダリング（バックエンド固有の実装が必要）
    fn render_atomic_max(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String;

    /// ベクトルロードをレンダリング（デフォルトはreinterpret_cast）
    fn render_vector_load(&self, ptr_expr: &str, offset_expr: &str, dtype: &str) -> String {
        format!(
            "*reinterpret_cast<{} *>(&{}[{}])",
            dtype, ptr_expr, offset_expr
        )
    }

    /// ベクトルストアをレンダリング（デフォルトはreinterpret_cast）
    fn render_vector_store(
        &self,
        ptr_expr: &str,
        offset_expr: &str,
        value_expr: &str,
        dtype: &str,
    ) -> String {
        format!(
            "*reinterpret_cast<{} *>(&{}[{}]) = {}",
            dtype, ptr_expr, offset_expr, value_expr
        )
    }

    // ========== 共通実装（デフォルト実装） ==========

    /// インデント文字列を取得
    fn indent(&self) -> String {
        " ".repeat(self.indent_level() * self.indent_size())
    }

    /// インデントレベルを増加
    fn inc_indent(&mut self) {
        *self.indent_level_mut() += 1;
    }

    /// インデントレベルを減少
    fn dec_indent(&mut self) {
        let level = self.indent_level_mut();
        if *level > 0 {
            *level -= 1;
        }
    }

    /// リテラルをレンダリング
    fn render_literal(&self, lit: &Literal) -> String {
        match lit {
            Literal::Bool(v) => {
                // Boolはu8として表現: true = 1, false = 0
                if *v { "1".to_string() } else { "0".to_string() }
            }
            Literal::I8(v) => format!("{}", v),
            Literal::I16(v) => format!("{}", v),
            Literal::I32(v) => format!("{}", v),
            Literal::I64(v) => format!("{}", v),
            Literal::U8(v) => format!("{}u", v),
            Literal::U16(v) => format!("{}u", v),
            Literal::U32(v) => format!("{}u", v),
            Literal::U64(v) => format!("{}ull", v),
            Literal::F32(v) => {
                // 特殊な浮動小数点値の処理
                if v.is_nan() {
                    "NAN".to_string()
                } else if v.is_infinite() {
                    if v.is_sign_positive() {
                        "INFINITY".to_string()
                    } else {
                        "(-INFINITY)".to_string()
                    }
                } else {
                    let s = format!("{}", v);
                    // 小数点が含まれていない場合は .0 を追加（0f → 0.0f）
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        format!("{}.0f", s)
                    } else {
                        format!("{}f", s)
                    }
                }
            }
            Literal::F64(v) => {
                if v.is_nan() {
                    "NAN".to_string()
                } else if v.is_infinite() {
                    if v.is_sign_positive() {
                        "INFINITY".to_string()
                    } else {
                        "(-INFINITY)".to_string()
                    }
                } else {
                    let s = format!("{}", v);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        format!("{}.0", s)
                    } else {
                        s
                    }
                }
            }
            Literal::F16(v) => {
                // F16はf32に変換してキャスト
                let f = v.to_f32();
                if f.is_nan() {
                    "(half)NAN".to_string()
                } else if f.is_infinite() {
                    if f.is_sign_positive() {
                        "(half)INFINITY".to_string()
                    } else {
                        "(half)(-INFINITY)".to_string()
                    }
                } else {
                    format!("(half){}f", f)
                }
            }
            Literal::BF16(v) => {
                // BF16はf32に変換してキャスト
                let f = v.to_f32();
                if f.is_nan() {
                    "(bfloat)NAN".to_string()
                } else if f.is_infinite() {
                    if f.is_sign_positive() {
                        "(bfloat)INFINITY".to_string()
                    } else {
                        "(bfloat)(-INFINITY)".to_string()
                    }
                } else {
                    format!("(bfloat){}f", f)
                }
            }
            Literal::Complex32(re, im) => {
                // Complex32は構造体リテラルとしてレンダリング
                format!("(complex32){{{}f, {}f}}", re, im)
            }
            Literal::Complex64(re, im) => {
                // Complex64は構造体リテラルとしてレンダリング
                format!("(complex64){{{}, {}}}", re, im)
            }
        }
    }

    /// AstNodeを式として描画
    fn render_expr(&self, node: &AstNode) -> String {
        match node {
            AstNode::Wildcard(name) => name.clone(),
            AstNode::Const(lit) => self.render_literal(lit),
            AstNode::Var(name) => name.clone(),
            AstNode::Add(left, right) => {
                format!("({} + {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Mul(left, right) => {
                format!("({} * {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Max(left, right) => {
                let args = vec![self.render_expr(left), self.render_expr(right)];
                self.render_math_func("max", &args)
            }
            AstNode::Rem(left, right) => {
                format!("({} % {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Idiv(left, right) => {
                format!("({} / {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Recip(operand) => {
                format!("(1.0f / {})", self.render_expr(operand))
            }
            AstNode::Sqrt(operand) => {
                let args = vec![self.render_expr(operand)];
                self.render_math_func("sqrt", &args)
            }
            AstNode::Log2(operand) => {
                let args = vec![self.render_expr(operand)];
                self.render_math_func("log2", &args)
            }
            AstNode::Exp2(operand) => {
                let args = vec![self.render_expr(operand)];
                self.render_math_func("exp2", &args)
            }
            AstNode::Sin(operand) => {
                let args = vec![self.render_expr(operand)];
                self.render_math_func("sin", &args)
            }
            AstNode::Floor(operand) => {
                let args = vec![self.render_expr(operand)];
                self.render_math_func("floor", &args)
            }
            // 0〜1の一様乱数を生成
            AstNode::Rand => "((float)rand() / (float)RAND_MAX)".to_string(),
            AstNode::BitwiseAnd(left, right) => {
                format!("({} & {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::BitwiseOr(left, right) => {
                format!("({} | {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::BitwiseXor(left, right) => {
                format!("({} ^ {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::BitwiseNot(operand) => {
                format!("(~{})", self.render_expr(operand))
            }
            AstNode::LeftShift(left, right) => {
                format!(
                    "({} << {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            AstNode::RightShift(left, right) => {
                format!(
                    "({} >> {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            // Comparison operations (primitive: Lt only)
            AstNode::Lt(left, right) => {
                format!("({} < {})", self.render_expr(left), self.render_expr(right))
            }
            // Logical operations (primitives: And, Not)
            AstNode::And(left, right) => {
                format!(
                    "({} && {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            AstNode::Not(operand) => {
                format!("(!{})", self.render_expr(operand))
            }
            // Select (ternary conditional)
            AstNode::Select {
                cond,
                then_val,
                else_val,
            } => {
                format!(
                    "({} ? {} : {})",
                    self.render_expr(cond),
                    self.render_expr(then_val),
                    self.render_expr(else_val)
                )
            }
            AstNode::Cast(operand, dtype) => {
                // C-style cast: (type)(value)
                format!(
                    "({})({})",
                    self.render_dtype_backend(dtype),
                    self.render_expr(operand)
                )
            }
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => {
                if *count == 1 {
                    format!("{}[{}]", self.render_expr(ptr), self.render_expr(offset))
                } else {
                    // ベクトルロード
                    let ptr_expr = self.render_expr(ptr);
                    let offset_expr = self.render_expr(offset);
                    let dtype_str = self.render_dtype_backend(dtype);
                    self.render_vector_load(&ptr_expr, &offset_expr, &dtype_str)
                }
            }
            AstNode::Call { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.render_expr(a)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            AstNode::Return { value } => {
                format!("return {}", self.render_expr(value))
            }
            AstNode::Allocate { dtype, size } => {
                let dtype_str = self.render_dtype_backend(dtype);
                let size_expr = self.render_expr(size);
                format!(
                    "({}*)malloc({} * sizeof({}))",
                    dtype_str, size_expr, dtype_str
                )
            }
            AstNode::Fma { a, b, c } => self.render_fma(
                &self.render_expr(a),
                &self.render_expr(b),
                &self.render_expr(c),
            ),
            AstNode::AtomicAdd {
                ptr,
                offset,
                value,
                dtype,
            } => self.render_atomic_add(
                &self.render_expr(ptr),
                &self.render_expr(offset),
                &self.render_expr(value),
                dtype,
            ),
            AstNode::AtomicMax {
                ptr,
                offset,
                value,
                dtype,
            } => self.render_atomic_max(
                &self.render_expr(ptr),
                &self.render_expr(offset),
                &self.render_expr(value),
                dtype,
            ),
            // Complex number operations
            AstNode::Real(operand) => {
                // Extract real part: z.re
                format!("({}).re", self.render_expr(operand))
            }
            AstNode::Imag(operand) => {
                // Extract imaginary part: z.im
                format!("({}).im", self.render_expr(operand))
            }
            AstNode::Conj(operand) => {
                // Complex conjugate: conj(z) = z.re - z.im*i
                // Render as struct literal with negated imaginary part
                let z = self.render_expr(operand);
                format!("((typeof({0})){{({0}).re, -({0}).im}})", z)
            }
            AstNode::MakeComplex { re, im } => {
                // Construct complex from real and imaginary parts
                // We don't know the type here, so use a generic approach
                // The caller should cast if needed
                format!(
                    "(complex32){{(float)({}), (float)({})}}",
                    self.render_expr(re),
                    self.render_expr(im)
                )
            }
            _ => format!("/* unsupported expression: {:?} */", node),
        }
    }

    /// 文として描画
    fn render_statement(&mut self, node: &AstNode) -> String {
        match node {
            AstNode::Store { ptr, offset, value } => {
                let value_type = value.infer_type();
                if value_type.is_vec() {
                    // ベクトルストア
                    let ptr_expr = self.render_expr(ptr);
                    let offset_expr = self.render_expr(offset);
                    let value_expr = self.render_expr(value);
                    let dtype_str = self.render_dtype_backend(&value_type);
                    format!(
                        "{}{};",
                        self.indent(),
                        self.render_vector_store(&ptr_expr, &offset_expr, &value_expr, &dtype_str)
                    )
                } else {
                    // スカラーストア
                    format!(
                        "{}{}[{}] = {};",
                        self.indent(),
                        self.render_expr(ptr),
                        self.render_expr(offset),
                        self.render_expr(value)
                    )
                }
            }
            AstNode::Assign { var, value } => {
                // 単なる代入として扱う（変数宣言はBlockで行われる）
                format!("{}{} = {};", self.indent(), var, self.render_expr(value))
            }
            AstNode::Block { statements, scope } => self.render_block(statements, scope),
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
                parallel,
            } => self.render_range(var, start, step, stop, body, parallel),
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => self.render_if(condition, then_body, else_body.as_deref()),
            AstNode::Return { value } => {
                format!("{}return {};", self.indent(), self.render_expr(value))
            }
            AstNode::Barrier => {
                format!("{}{}", self.indent(), self.render_barrier_backend())
            }
            AstNode::Call { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.render_expr(a)).collect();
                format!("{}{}({});", self.indent(), name, arg_strs.join(", "))
            }
            AstNode::CallKernel {
                name,
                args,
                grid_size,
                thread_group_size,
            } => {
                // カーネル呼び出しをレンダリング（dispatch情報をコメントで出力）
                let arg_strs: Vec<String> = args.iter().map(|a| self.render_expr(a)).collect();
                let grid = format!(
                    "({}, {}, {})",
                    self.render_expr(&grid_size[0]),
                    self.render_expr(&grid_size[1]),
                    self.render_expr(&grid_size[2])
                );
                let tg = format!(
                    "({}, {}, {})",
                    self.render_expr(&thread_group_size[0]),
                    self.render_expr(&thread_group_size[1]),
                    self.render_expr(&thread_group_size[2])
                );
                format!(
                    "{}// dispatch: grid={}, thread_group={}\n{}{}({});",
                    self.indent(),
                    grid,
                    tg,
                    self.indent(),
                    name,
                    arg_strs.join(", ")
                )
            }
            AstNode::Deallocate { ptr } => {
                format!("{}free({});", self.indent(), self.render_expr(ptr))
            }
            _ => {
                // 式として評価できるものは文末にセミコロンを付ける
                format!("{}{};", self.indent(), self.render_expr(node))
            }
        }
    }

    /// ブロックを描画
    fn render_block(&mut self, statements: &[AstNode], scope: &crate::ast::Scope) -> String {
        let mut result = String::new();

        // ブロック先頭で変数宣言を出力（初期値なし）
        for var_decl in scope.local_variables() {
            let type_str = self.render_dtype_backend(&var_decl.dtype);
            result.push_str(&format!(
                "{}{} {};\n",
                self.indent(),
                type_str,
                var_decl.name
            ));
        }

        // 文を描画
        for stmt in statements {
            result.push_str(&self.render_statement(stmt));
            result.push('\n');
        }
        result
    }

    /// Rangeループを描画
    fn render_range(
        &mut self,
        var: &str,
        start: &AstNode,
        step: &AstNode,
        stop: &AstNode,
        body: &AstNode,
        _parallel: &crate::ast::ParallelInfo, // 将来のOpenMPサポート用
    ) -> String {
        let mut result = String::new();
        let loop_var_type = self.render_dtype_backend(&DType::I64);
        result.push_str(&format!(
            "{}for ({} {} = {}; {} < {}; {} += {}) {{",
            self.indent(),
            loop_var_type,
            var,
            self.render_expr(start),
            var,
            self.render_expr(stop),
            var,
            self.render_expr(step)
        ));
        result.push('\n');
        self.inc_indent();
        let body_str = self.render_statement(body);
        result.push_str(&body_str);
        // Blockでない場合は改行が含まれていないので追加
        if !body_str.ends_with('\n') {
            result.push('\n');
        }
        self.dec_indent();
        result.push_str(&format!("{}}}", self.indent()));
        result
    }

    /// If文を描画
    fn render_if(
        &mut self,
        condition: &AstNode,
        then_body: &AstNode,
        else_body: Option<&AstNode>,
    ) -> String {
        let mut result = String::new();
        result.push_str(&format!(
            "{}if ({}) {{",
            self.indent(),
            self.render_expr(condition)
        ));
        result.push('\n');
        self.inc_indent();
        let then_str = self.render_statement(then_body);
        result.push_str(&then_str);
        if !then_str.ends_with('\n') {
            result.push('\n');
        }
        self.dec_indent();
        result.push_str(&format!("{}}}", self.indent()));

        if let Some(else_b) = else_body {
            result.push_str(" else {\n");
            self.inc_indent();
            let else_str = self.render_statement(else_b);
            result.push_str(&else_str);
            if !else_str.ends_with('\n') {
                result.push('\n');
            }
            self.dec_indent();
            result.push_str(&format!("{}}}", self.indent()));
        }

        result
    }

    /// 関数パラメータを描画
    fn render_param(&self, param: &VarDecl, is_kernel: bool) -> String {
        let attribute = self.render_param_attribute(param, is_kernel);
        if attribute.is_empty() {
            // パラメータとして含めない（OpenMPのGroupIdなど）
            String::new()
        } else {
            attribute
        }
    }

    /// プログラム全体を描画（デフォルト実装）
    ///
    /// カーネル関数群のみを出力します。
    /// カーネルの実行順序はホスト側（CompiledProgram）で管理されます。
    fn render_program_clike(&mut self, program: &AstNode) -> String {
        let AstNode::Program { functions, .. } = program else {
            panic!("Expected AstNode::Program");
        };

        let mut result = String::new();

        // ヘッダー
        result.push_str(&self.render_header());

        // カーネル関数（Kernel）とその他の関数（Function）を分離
        let mut kernel_functions: Vec<_> = Vec::new();
        let mut helper_functions: Vec<_> = Vec::new();

        // 関数名を取得するヘルパー関数
        fn get_func_name(func: &AstNode) -> Option<String> {
            match func {
                AstNode::Function { name, .. } => name.clone(),
                AstNode::Kernel { name, .. } => name.clone(),
                _ => None,
            }
        }

        for func in functions {
            if matches!(func, AstNode::Kernel { .. }) {
                kernel_functions.push(func);
            } else if matches!(func, AstNode::Function { .. }) {
                helper_functions.push(func);
            }
        }

        // カーネル関数を名前でソート
        kernel_functions.sort_by_key(|func| get_func_name(func).unwrap_or_default());

        // ヘルパー関数を名前でソート
        helper_functions.sort_by_key(|func| get_func_name(func).unwrap_or_default());

        // カーネル関数を最初に描画
        if !kernel_functions.is_empty() {
            result.push_str("// === Kernel Functions ===\n");
            for func in kernel_functions {
                result.push_str(&self.render_function_node(func));
                result.push('\n');
            }
        }

        // ヘルパー関数を描画
        if !helper_functions.is_empty() {
            result.push_str("// === Helper Functions ===\n");
            for func in helper_functions {
                result.push_str(&self.render_function_node(func));
                result.push('\n');
            }
        }

        result
    }

    /// AstNode::FunctionまたはAstNode::Kernelをレンダリング
    fn render_function_node(&mut self, func_node: &AstNode) -> String {
        // FunctionとKernelの両方を処理
        let (name, params, return_type, body, is_kernel) = match func_node {
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => (name, params, return_type, body, false),
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                ..
            } => (name, params, return_type, body, true),
            _ => panic!("Expected AstNode::Function or AstNode::Kernel"),
        };

        let func_name = name.as_ref().map(|s| s.as_str()).unwrap_or("anonymous");

        let mut result = String::new();

        // 関数修飾子（kernel, __globalなど）
        let qualifier = self.render_function_qualifier(is_kernel);
        if !qualifier.is_empty() {
            result.push_str(&qualifier);
            result.push(' ');
        }

        // 返り値の型
        result.push_str(&self.render_dtype_backend(return_type));
        result.push(' ');

        // 関数名
        result.push_str(func_name);
        result.push('(');

        // パラメータリスト（空文字列のパラメータはスキップ）
        let mut rendered_params: Vec<String> = params
            .iter()
            .map(|p| self.render_param(p, is_kernel))
            .filter(|s| !s.is_empty())
            .collect();

        // カーネル関数の場合、追加パラメータ（Metal用のuint3 _gidなど）を追加
        if is_kernel {
            let extra_params = self.render_extra_kernel_params(params);
            rendered_params.extend(extra_params);
        }

        result.push_str(&rendered_params.join(", "));

        result.push_str(") {\n");

        // 関数本体（Blockノードのはず）
        self.inc_indent();

        // スレッドID等の特殊変数の宣言（カーネル関数の場合）
        let thread_vars = self.render_thread_var_declarations(params, &self.indent());
        if !thread_vars.is_empty() {
            result.push_str(&thread_vars);
        }

        let body_str = self.render_statement(body);
        result.push_str(&body_str);
        // Blockでない場合は改行が含まれていないので追加
        if !body_str.ends_with('\n') {
            result.push('\n');
        }
        self.dec_indent();
        result.push_str(&format!("{}}}\n", self.indent()));
        result
    }
}

/// ASTノードからバッファプレースホルダー（input0, input1, ..., output）を抽出
///
/// 関数本体を走査して、`inputN`パターンの変数と`output`変数を見つけます。
/// これはlowering時にパラメータリストが空のまま生成されるFunctionノードに対して
/// バッファパラメータを自動生成するために使用されます。
pub fn extract_buffer_placeholders(body: &AstNode) -> (Vec<String>, bool) {
    use std::collections::HashSet;

    let mut inputs: HashSet<String> = HashSet::new();
    let mut has_output = false;

    fn visit(node: &AstNode, inputs: &mut HashSet<String>, has_output: &mut bool) {
        match node {
            AstNode::Var(name) => {
                if name == "output" {
                    *has_output = true;
                } else if name.starts_with("input") {
                    inputs.insert(name.clone());
                }
            }
            AstNode::Load { ptr, offset, .. } => {
                visit(ptr, inputs, has_output);
                visit(offset, inputs, has_output);
            }
            AstNode::Store { ptr, offset, value } => {
                visit(ptr, inputs, has_output);
                visit(offset, inputs, has_output);
                visit(value, inputs, has_output);
            }
            // 算術演算（2項）
            AstNode::Add(l, r)
            | AstNode::Mul(l, r)
            | AstNode::Max(l, r)
            | AstNode::Rem(l, r)
            | AstNode::Idiv(l, r)
            | AstNode::Lt(l, r)
            | AstNode::And(l, r)
            | AstNode::BitwiseAnd(l, r)
            | AstNode::BitwiseOr(l, r)
            | AstNode::BitwiseXor(l, r)
            | AstNode::LeftShift(l, r)
            | AstNode::RightShift(l, r) => {
                visit(l, inputs, has_output);
                visit(r, inputs, has_output);
            }
            // 算術演算（1項）
            AstNode::Recip(x)
            | AstNode::Sqrt(x)
            | AstNode::Log2(x)
            | AstNode::Exp2(x)
            | AstNode::Sin(x)
            | AstNode::Floor(x)
            | AstNode::BitwiseNot(x)
            | AstNode::Not(x) => {
                visit(x, inputs, has_output);
            }
            // Select (ternary)
            AstNode::Select {
                cond,
                then_val,
                else_val,
            } => {
                visit(cond, inputs, has_output);
                visit(then_val, inputs, has_output);
                visit(else_val, inputs, has_output);
            }
            AstNode::Cast(x, _) => {
                visit(x, inputs, has_output);
            }
            AstNode::Call { args, .. } => {
                for arg in args {
                    visit(arg, inputs, has_output);
                }
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    visit(stmt, inputs, has_output);
                }
            }
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                visit(condition, inputs, has_output);
                visit(then_body, inputs, has_output);
                if let Some(else_b) = else_body {
                    visit(else_b, inputs, has_output);
                }
            }
            AstNode::Range {
                start,
                step,
                stop,
                body,
                ..
            } => {
                visit(start, inputs, has_output);
                visit(step, inputs, has_output);
                visit(stop, inputs, has_output);
                visit(body, inputs, has_output);
            }
            AstNode::Assign { value, .. } => {
                visit(value, inputs, has_output);
            }
            AstNode::Return { value } => {
                visit(value, inputs, has_output);
            }
            AstNode::Allocate { size, .. } => {
                visit(size, inputs, has_output);
            }
            AstNode::Deallocate { ptr } => {
                visit(ptr, inputs, has_output);
            }
            AstNode::Fma { a, b, c } => {
                visit(a, inputs, has_output);
                visit(b, inputs, has_output);
                visit(c, inputs, has_output);
            }
            _ => {}
        }
    }

    visit(body, &mut inputs, &mut has_output);

    // inputNをソートして返す（input0, input1, input2...の順）
    let mut input_vec: Vec<String> = inputs.into_iter().collect();
    input_vec.sort_by(|a, b| {
        let a_num: usize = a.strip_prefix("input").unwrap_or("0").parse().unwrap_or(0);
        let b_num: usize = b.strip_prefix("input").unwrap_or("0").parse().unwrap_or(0);
        a_num.cmp(&b_num)
    });

    (input_vec, has_output)
}

// ============================================================================
// Generic C-like Renderer (Default Implementation)
// ============================================================================

/// デフォルトのC言語ライクなレンダラー
///
/// OpenCLやMetalなどの特殊なバックエンドを使用できない場合のフォールバックとして使用できます。
/// 標準的なC言語構文で出力しますが、カーネル実行は行えません（コード生成のみ）。
#[derive(Debug, Clone, Default)]
pub struct GenericRenderer {
    indent_level: usize,
}

impl GenericRenderer {
    /// 新しいGenericRendererを作成
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }
}

impl Renderer for GenericRenderer {
    type CodeRepr = String;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut r = self.clone();
        r.render_program_clike(program)
    }

    fn is_available(&self) -> bool {
        true
    }
}

impl CLikeRenderer for GenericRenderer {
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
            DType::Complex32 => "complex32".to_string(),
            DType::Complex64 => "complex64".to_string(),
            DType::Int => "long long".to_string(), // Index type: 64-bit for CPU
            DType::Ptr(inner) => format!("{}*", self.render_dtype_backend(inner)),
            DType::Vec(inner, size) => format!("{}[{}]", self.render_dtype_backend(inner), size),
            DType::Tuple(_) => "/* tuple */".to_string(),
            DType::Unknown => "/* unknown */".to_string(),
        }
    }

    fn render_barrier_backend(&self) -> String {
        "// barrier (no-op in generic C)".to_string()
    }

    fn render_header(&self) -> String {
        r#"// Generated C-like code (generic backend)
#include <math.h>
#include <stdlib.h>

// Complex number types (interleaved layout)
typedef struct { float re; float im; } complex32;
typedef struct { double re; double im; } complex64;

"#
        .to_string()
    }

    fn render_function_qualifier(&self, _is_kernel: bool) -> String {
        String::new()
    }

    fn render_param_attribute(&self, param: &VarDecl, _is_kernel: bool) -> String {
        // 特殊なパラメータ（スレッドIDなど）は通常の関数では不要
        match param.name.as_str() {
            "group_id_x" | "group_id_y" | "group_id_z" | "local_id_x" | "local_id_y"
            | "local_id_z" | "global_id_x" | "global_id_y" | "global_id_z" => {
                // スレッドID等は引数として渡す
                let type_str = self.render_dtype_backend(&param.dtype);
                format!("{} {}", type_str, param.name)
            }
            _ => {
                let type_str = self.render_dtype_backend(&param.dtype);
                format!("{} {}", type_str, param.name)
            }
        }
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
            _ => format!("{}({})", name, args.join(", ")),
        }
    }

    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, _dtype: &DType) -> String {
        // Generic Cではアトミック操作をシミュレート（警告付き）
        format!(
            "/* WARNING: non-atomic */ ({}[{}] += {})",
            ptr, offset, value
        )
    }

    fn render_atomic_max(&self, ptr: &str, offset: &str, value: &str, _dtype: &DType) -> String {
        // Generic Cではアトミック操作をシミュレート（警告付き）
        format!(
            "/* WARNING: non-atomic */ ({}[{}] = fmaxf({}[{}], {}))",
            ptr, offset, ptr, offset, value
        )
    }
}
