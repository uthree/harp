use crate::ast::{AstNode, DType, FunctionKind, Literal, VarDecl};
use crate::backend::Renderer;

// C言語に近い構文の言語のためのレンダラー
// Metal, CUDA, C(with OpenMP), OpenCLなどのバックエンドは大体C言語に近い文法を採用しているので、共通化したい。

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
    fn render_function_qualifier(&self, func_kind: &FunctionKind) -> String;

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
            Literal::F32(v) => format!("{}f", v),
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
            _ => {
                // 式として評価できるものは文末にセミコロンを付ける
                format!("{}{};", self.indent(), self.render_expr(node))
            }
        }
    }

    /// ブロックを描画
    fn render_block(&mut self, statements: &[AstNode], scope: &crate::ast::Scope) -> String {
        let mut result = String::new();

        // ブロック先頭で変数宣言を出力
        for var_decl in scope.local_variables() {
            if let Some(initial_value) = &var_decl.initial_value {
                let type_str = self.render_dtype_backend(&var_decl.dtype);
                result.push_str(&format!(
                    "{}{} {} = {};\n",
                    self.indent(),
                    type_str,
                    var_decl.name,
                    self.render_expr(initial_value)
                ));
            }
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
    fn render_program_clike(&mut self, program: &AstNode) -> String {
        let AstNode::Program {
            functions,
            entry_point,
        } = program
        else {
            panic!("Expected AstNode::Program");
        };

        let mut result = String::new();

        // ヘッダー
        result.push_str(&self.render_header());

        // エントリーポイント情報をコメントとして追加
        result.push_str(&format!("// Entry Point: {}\n\n", entry_point));

        // カーネル関数（kernel_*）とその他の関数を分離
        let mut kernel_functions: Vec<_> = Vec::new();
        let mut other_functions: Vec<_> = Vec::new();
        let mut entry_func: Option<&AstNode> = None;

        for func in functions {
            if let AstNode::Function {
                name: Some(name), ..
            } = func
            {
                if name == entry_point {
                    entry_func = Some(func);
                } else if name.starts_with("kernel_") {
                    kernel_functions.push(func);
                } else {
                    other_functions.push(func);
                }
            }
        }

        // カーネル関数を番号順にソート
        kernel_functions.sort_by_key(|func| {
            if let AstNode::Function {
                name: Some(name), ..
            } = func
            {
                name.strip_prefix("kernel_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(usize::MAX)
            } else {
                usize::MAX
            }
        });

        // その他の関数を名前順にソート
        other_functions.sort_by_key(|func| {
            if let AstNode::Function {
                name: Some(name), ..
            } = func
            {
                name.clone()
            } else {
                String::new()
            }
        });

        // カーネル関数を最初に描画
        if !kernel_functions.is_empty() {
            result.push_str("// === Kernel Functions ===\n");
            for func in kernel_functions {
                result.push_str(&self.render_function_node(func));
                result.push('\n');
            }
        }

        // その他の関数を描画
        if !other_functions.is_empty() {
            result.push_str("// === Helper Functions ===\n");
            for func in other_functions {
                result.push_str(&self.render_function_node(func));
                result.push('\n');
            }
        }

        // エントリーポイント関数を最後に描画
        if let Some(func) = entry_func {
            result.push_str("// === Entry Point Function ===\n");
            result.push_str(&self.render_function_node(func));
            result.push('\n');
        }

        result
    }

    /// AstNode::Functionをレンダリング
    fn render_function_node(&mut self, func_node: &AstNode) -> String {
        if let AstNode::Function {
            name,
            params,
            return_type,
            body,
            kind,
        } = func_node
        {
            let func_name = name.as_ref().map(|s| s.as_str()).unwrap_or("anonymous");

            let mut result = String::new();

            // 関数修飾子（kernel, __globalなど）
            let qualifier = self.render_function_qualifier(kind);
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

            // パラメータリスト
            let is_kernel = matches!(kind, FunctionKind::Kernel(_));
            for (i, param) in params.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push_str(&self.render_param(param, is_kernel));
            }

            result.push_str(") {\n");

            // 関数本体（Blockノードのはず）
            self.inc_indent();

            // スレッドID等の特殊変数の宣言（カーネル関数の場合）
            let thread_vars = self.render_thread_var_declarations(params, &self.indent());
            if !thread_vars.is_empty() {
                result.push_str(&thread_vars);
            }

            result.push_str(&self.render_statement(body));
            self.dec_indent();
            result.push_str(&format!("{}}}\n", self.indent()));

            result
        } else {
            panic!("Expected AstNode::Function");
        }
    }
}
