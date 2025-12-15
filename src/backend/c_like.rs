use crate::ast::{AstNode, DType, Literal, VarDecl};
use crate::backend::Renderer;

// C言語に近い構文の言語のためのレンダラー
// Metal, CUDA, OpenCLなどのバックエンドは大体C言語に近い文法を採用しているので、共通化したい。

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

pub trait CLikeRenderer: Renderer {
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

    /// 数学関数をレンダリング（max vs fmaxf など）
    fn render_math_func(&self, name: &str, args: &[String]) -> String;

    /// ベクトルロードをレンダリング（デフォルトはreinterpret_cast）
    fn render_vector_load(&self, ptr_expr: &str, offset_expr: &str, dtype: &str) -> String {
        format!(
            "*reinterpret_cast<{} *>(&{}[{}])",
            dtype, ptr_expr, offset_expr
        )
    }

    /// libloading用のラッパー関数名を返す
    fn libloading_wrapper_name(&self) -> &'static str;

    /// libloading用のラッパー関数を生成
    ///
    /// libloadingは固定シグネチャ `void f(void** buffers)` を期待するため、
    /// エントリーポイント関数をラップする関数を生成する。
    fn render_libloading_wrapper(&self, entry_func: &AstNode, entry_point: &str) -> String;

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
            Literal::F32(v) => {
                let s = format!("{}", v);
                // 小数点が含まれていない場合は .0 を追加（0f → 0.0f）
                if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    format!("{}.0f", s)
                } else {
                    format!("{}f", s)
                }
            }
            Literal::Int(v) => format!("{}", v),
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
            // Comparison operations
            AstNode::Lt(left, right) => {
                format!("({} < {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Le(left, right) => {
                format!(
                    "({} <= {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            AstNode::Gt(left, right) => {
                format!("({} > {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Ge(left, right) => {
                format!(
                    "({} >= {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            AstNode::Eq(left, right) => {
                format!(
                    "({} == {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            AstNode::Ne(left, right) => {
                format!(
                    "({} != {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            AstNode::Cast(operand, dtype) => {
                format!(
                    "{}({})",
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
            _ => format!("/* unsupported expression: {:?} */", node),
        }
    }

    /// 文として描画
    fn render_statement(&mut self, node: &AstNode) -> String {
        match node {
            AstNode::Store { ptr, offset, value } => {
                format!(
                    "{}{}[{}] = {};",
                    self.indent(),
                    self.render_expr(ptr),
                    self.render_expr(offset),
                    self.render_expr(value)
                )
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
            } => self.render_range(var, start, step, stop, body),
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
    ) -> String {
        let mut result = String::new();
        let loop_var_type = self.render_dtype_backend(&DType::Int);
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
            // パラメータとして含めない（OpenMPのThreadIdなど）
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
        let AstNode::Program { functions } = program else {
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
        let rendered_params: Vec<String> = params
            .iter()
            .map(|p| self.render_param(p, is_kernel))
            .filter(|s| !s.is_empty())
            .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Literal, Mutability, Scope};
    use crate::backend::opencl::OpenCLRenderer;

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_render_literal() {
        let renderer = OpenCLRenderer::new();

        // 整数
        assert_eq!(renderer.render_literal(&Literal::Int(42)), "42");

        // 浮動小数点（小数点あり）
        assert_eq!(renderer.render_literal(&Literal::F32(3.14)), "3.14f");

        // 浮動小数点（小数点なし → .0が追加される）
        assert_eq!(renderer.render_literal(&Literal::F32(5.0)), "5.0f");
    }

    #[test]
    fn test_render_expr_bitwise_operations() {
        let renderer = OpenCLRenderer::new();

        // BitwiseAnd
        let and_expr = AstNode::BitwiseAnd(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        assert_eq!(renderer.render_expr(&and_expr), "(a & b)");

        // BitwiseOr
        let or_expr = AstNode::BitwiseOr(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        assert_eq!(renderer.render_expr(&or_expr), "(a | b)");

        // BitwiseXor
        let xor_expr = AstNode::BitwiseXor(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        assert_eq!(renderer.render_expr(&xor_expr), "(a ^ b)");

        // BitwiseNot
        let not_expr = AstNode::BitwiseNot(Box::new(AstNode::Var("x".to_string())));
        assert_eq!(renderer.render_expr(&not_expr), "(~x)");
    }

    #[test]
    fn test_render_expr_shift_operations() {
        let renderer = OpenCLRenderer::new();

        // LeftShift
        let left_shift = AstNode::LeftShift(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(2))),
        );
        assert_eq!(renderer.render_expr(&left_shift), "(x << 2)");

        // RightShift
        let right_shift = AstNode::RightShift(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(3))),
        );
        assert_eq!(renderer.render_expr(&right_shift), "(x >> 3)");
    }

    #[test]
    fn test_render_expr_math_operations() {
        let renderer = OpenCLRenderer::new();

        // Recip
        let recip = AstNode::Recip(Box::new(AstNode::Var("x".to_string())));
        assert_eq!(renderer.render_expr(&recip), "(1.0f / x)");

        // Rem (modulo)
        let rem = AstNode::Rem(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        assert_eq!(renderer.render_expr(&rem), "(a % b)");

        // Idiv
        let idiv = AstNode::Idiv(
            Box::new(AstNode::Var("a".to_string())),
            Box::new(AstNode::Var("b".to_string())),
        );
        assert_eq!(renderer.render_expr(&idiv), "(a / b)");
    }

    #[test]
    fn test_render_expr_cast() {
        let renderer = OpenCLRenderer::new();

        let cast = AstNode::Cast(Box::new(AstNode::Var("x".to_string())), DType::Int);
        assert_eq!(renderer.render_expr(&cast), "int(x)");
    }

    #[test]
    fn test_render_expr_load() {
        let renderer = OpenCLRenderer::new();

        // スカラロード
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Var("arr".to_string())),
            offset: Box::new(AstNode::Var("i".to_string())),
            count: 1,
            dtype: DType::F32,
        };
        assert_eq!(renderer.render_expr(&load), "arr[i]");
    }

    #[test]
    fn test_render_expr_call() {
        let renderer = OpenCLRenderer::new();

        let call = AstNode::Call {
            name: "foo".to_string(),
            args: vec![
                AstNode::Var("a".to_string()),
                AstNode::Const(Literal::Int(42)),
            ],
        };
        assert_eq!(renderer.render_expr(&call), "foo(a, 42)");
    }

    #[test]
    fn test_render_expr_return() {
        let renderer = OpenCLRenderer::new();

        let ret = AstNode::Return {
            value: Box::new(AstNode::Var("result".to_string())),
        };
        assert_eq!(renderer.render_expr(&ret), "return result");
    }

    #[test]
    fn test_render_statement_store() {
        let mut renderer = OpenCLRenderer::new();

        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var("arr".to_string())),
            offset: Box::new(AstNode::Var("i".to_string())),
            value: Box::new(AstNode::Const(Literal::F32(1.0))),
        };
        let rendered = renderer.render_statement(&store);
        assert!(rendered.contains("arr[i] = 1.0f;"));
    }

    #[test]
    fn test_render_statement_assign() {
        let mut renderer = OpenCLRenderer::new();

        let assign = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Const(Literal::Int(10))),
        };
        let rendered = renderer.render_statement(&assign);
        assert!(rendered.contains("x = 10;"));
    }

    #[test]
    fn test_render_range_loop() {
        let mut renderer = OpenCLRenderer::new();

        let scope = Scope::new();
        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body: Box::new(AstNode::Block {
                statements: vec![],
                scope: Box::new(scope),
            }),
        };

        let rendered = renderer.render_statement(&range);
        assert!(rendered.contains("for (int i = 0; i < 10; i += 1)"));
    }

    #[test]
    fn test_render_block_with_local_variables() {
        let mut renderer = OpenCLRenderer::new();

        let mut scope = Scope::new();
        scope
            .declare("temp".to_string(), DType::F32, Mutability::Mutable)
            .unwrap();

        let block = AstNode::Block {
            statements: vec![AstNode::Assign {
                var: "temp".to_string(),
                value: Box::new(AstNode::Const(Literal::F32(0.0))),
            }],
            scope: Box::new(scope),
        };

        let rendered = renderer.render_statement(&block);
        // ローカル変数の宣言が含まれること
        assert!(rendered.contains("float temp;"));
        // 代入文が含まれること
        assert!(rendered.contains("temp = 0.0f;"));
    }

    #[test]
    fn test_render_barrier_statement() {
        let mut renderer = OpenCLRenderer::new();
        let barrier = AstNode::Barrier;
        let rendered = renderer.render_statement(&barrier);
        // 重要なのは、エラーなくレンダリングされること
        // OpenCLRendererは "barrier(..)" を出力する
        assert!(rendered.contains("barrier"));
    }

    #[test]
    fn test_indent_management() {
        let mut renderer = OpenCLRenderer::new();
        assert_eq!(renderer.indent_level(), 0);

        renderer.inc_indent();
        assert_eq!(renderer.indent_level(), 1);
        assert_eq!(renderer.indent(), "    "); // 4 spaces

        renderer.inc_indent();
        assert_eq!(renderer.indent_level(), 2);
        assert_eq!(renderer.indent(), "        "); // 8 spaces

        renderer.dec_indent();
        assert_eq!(renderer.indent_level(), 1);

        renderer.dec_indent();
        assert_eq!(renderer.indent_level(), 0);

        // Should not go negative
        renderer.dec_indent();
        assert_eq!(renderer.indent_level(), 0);
    }
}
