use crate::ast::{
    AstNode, DType, Function, FunctionKind, Literal, Mutability, Program, VarDecl, VarKind,
};
use crate::backend::Renderer;
use crate::backend::openmp::CCode;

/// C言語とOpenMP用のレンダラー
#[derive(Debug, Clone)]
pub struct CRenderer {
    indent_level: usize,
    indent_size: usize,
}

impl CRenderer {
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            indent_size: 4,
        }
    }

    fn indent(&self) -> String {
        " ".repeat(self.indent_level * self.indent_size)
    }

    fn inc_indent(&mut self) {
        self.indent_level += 1;
    }

    fn dec_indent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    /// DTypeをC言語の型文字列に変換
    #[allow(clippy::only_used_in_recursion)]
    fn render_dtype(&self, dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "int".to_string(),
            DType::Usize => "unsigned int".to_string(),
            DType::Ptr(inner) => format!("{}*", self.render_dtype(inner)),
            DType::Vec(inner, size) => {
                // C言語ではベクトル型をサポートしていないので、配列として表現
                format!("{}[{}]", self.render_dtype(inner), size)
            }
            DType::Tuple(types) => {
                if types.is_empty() {
                    "void".to_string()
                } else {
                    // タプル型は構造体として表現
                    format!("tuple_{}", types.len())
                }
            }
            DType::Unknown => "auto".to_string(),
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
                format!(
                    "fmaxf({}, {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
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
                format!("sqrtf({})", self.render_expr(operand))
            }
            AstNode::Log2(operand) => {
                format!("log2f({})", self.render_expr(operand))
            }
            AstNode::Exp2(operand) => {
                format!("exp2f({})", self.render_expr(operand))
            }
            AstNode::Sin(operand) => {
                format!("sinf({})", self.render_expr(operand))
            }
            AstNode::Cast(operand, dtype) => {
                format!(
                    "({})({})",
                    self.render_dtype(dtype),
                    self.render_expr(operand)
                )
            }
            AstNode::Load { ptr, offset, count } => {
                if *count == 1 {
                    format!("{}[{}]", self.render_expr(ptr), self.render_expr(offset))
                } else {
                    // ベクトルロードは配列として扱う
                    format!("{}[{}]", self.render_expr(ptr), self.render_expr(offset))
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

    /// リテラルを描画
    fn render_literal(&self, lit: &Literal) -> String {
        match lit {
            Literal::F32(v) => format!("{}f", v),
            Literal::Isize(v) => format!("{}", v),
            Literal::Usize(v) => format!("{}u", v),
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
                // C言語では型推論がないので、型を明示的に書く必要がある
                // ただし、ここでは簡易的にfloatとして扱う
                let inferred_type = value.infer_type();
                let type_str = self.render_dtype(&inferred_type);
                format!(
                    "{}{} {} = {};",
                    self.indent(),
                    type_str,
                    var,
                    self.render_expr(value)
                )
            }
            AstNode::Block { statements, .. } => self.render_block(statements),
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
                // OpenMPのバリア
                format!("{}#pragma omp barrier", self.indent())
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
    fn render_block(&mut self, statements: &[AstNode]) -> String {
        let mut result = String::new();
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
        result.push_str(&format!(
            "{}for (unsigned int {} = {}; {} < {}; {} += {}) {{",
            self.indent(),
            var,
            self.render_expr(start),
            var,
            self.render_expr(stop),
            var,
            self.render_expr(step)
        ));
        result.push('\n');
        self.inc_indent();
        result.push_str(&self.render_statement(body));
        self.dec_indent();
        result.push_str(&format!("{}}}", self.indent()));
        result
    }

    /// 関数パラメータを描画
    fn render_param(&self, param: &VarDecl, _is_kernel: bool) -> String {
        let type_str = self.render_dtype(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        // ThreadId, GroupIdなどは関数内で取得する
        match &param.kind {
            VarKind::Normal => {
                format!("{}{} {}", mut_str, type_str, param.name)
            }
            VarKind::ThreadId(_)
            | VarKind::GroupId(_)
            | VarKind::GroupSize(_)
            | VarKind::GridSize(_) => {
                // これらはOpenMPのスレッドIDとして扱うため、パラメータには含めない
                // 関数内で omp_get_thread_num() などを使って取得する
                String::new()
            }
        }
    }

    /// 関数を描画
    fn render_function(&mut self, _name: &str, func: &Function) -> String {
        let mut result = String::new();

        // 返り値の型
        let return_type = self.render_dtype(&func.return_type);

        // パラメータリスト
        let is_kernel = matches!(func.kind, FunctionKind::Kernel(_));
        let params: Vec<String> = func
            .params
            .iter()
            .map(|p| self.render_param(p, is_kernel))
            .filter(|s| !s.is_empty())
            .collect();

        // 関数シグネチャ
        result.push_str(&format!("{}({}) {{", return_type, params.join(", ")));
        result.push('\n');

        // ThreadId, GroupIdなどの変数を関数の先頭で宣言
        self.inc_indent();
        for param in &func.params {
            match &param.kind {
                VarKind::ThreadId(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = omp_get_thread_num();\n",
                        self.indent(),
                        param.name
                    ));
                }
                VarKind::GroupId(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = 0; // group_id not supported\n",
                        self.indent(),
                        param.name
                    ));
                }
                VarKind::GroupSize(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = omp_get_num_threads();\n",
                        self.indent(),
                        param.name
                    ));
                }
                VarKind::GridSize(_) => {
                    result.push_str(&format!(
                        "{}unsigned int {} = 0; // grid_size not supported\n",
                        self.indent(),
                        param.name
                    ));
                }
                VarKind::Normal => {}
            }
        }

        // 関数本体
        result.push_str(&self.render_statement(&func.body));

        self.dec_indent();
        result.push_str("}\n");

        result
    }

    /// Programをレンダリング
    pub fn render_program(&mut self, program: &Program) -> CCode {
        let mut code = String::new();

        // ヘッダーファイルのインクルード
        code.push_str("#include <math.h>\n");
        code.push_str("#include <omp.h>\n");
        code.push_str("#include <stdint.h>\n");
        code.push('\n');

        // すべての関数を描画
        for (name, func) in &program.functions {
            code.push_str(&self.render_function(name, func));
            code.push('\n');
        }

        CCode::new(code)
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

    fn render(&self, program: &Program) -> Self::CodeRepr {
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
    use crate::ast::{DType, helper::*};

    #[test]
    fn test_render_dtype() {
        let renderer = CRenderer::new();
        assert_eq!(renderer.render_dtype(&DType::F32), "float");
        assert_eq!(renderer.render_dtype(&DType::Isize), "int");
        assert_eq!(renderer.render_dtype(&DType::Usize), "unsigned int");
        assert_eq!(
            renderer.render_dtype(&DType::Ptr(Box::new(DType::F32))),
            "float*"
        );
    }

    #[test]
    fn test_render_literal() {
        let renderer = CRenderer::new();
        assert_eq!(renderer.render_literal(&Literal::F32(1.5)), "1.5f");
        assert_eq!(renderer.render_literal(&Literal::Isize(42)), "42");
        assert_eq!(renderer.render_literal(&Literal::Usize(10)), "10u");
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
        use crate::ast::AccessRegion;

        let mut program = Program::new("test_entry".to_string());

        let func = Function::new(
            FunctionKind::Normal,
            vec![VarDecl {
                name: "x".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
                region: AccessRegion::Shared,
            }],
            DType::Tuple(vec![]),
            vec![store(
                var("x"),
                AstNode::Const(Literal::Usize(0)),
                AstNode::Const(Literal::F32(1.0)),
            )],
        )
        .unwrap();

        let _ = program.add_function("test_func".to_string(), func);

        let renderer = CRenderer::new();
        let code = renderer.render(&program);

        assert!(code.contains("#include <math.h>"));
        assert!(code.contains("#include <omp.h>"));
        assert!(code.contains("void("));
        assert!(code.contains("float* x"));
    }
}
