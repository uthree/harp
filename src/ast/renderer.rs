//! ASTを可読性の高い文字列に変換するモジュール

use super::{AstNode, Literal};

/// ASTを簡潔な文字列表現に変換するレンダラー
pub struct SimpleAstRenderer {
    /// インデントの深さ
    indent_level: usize,
    /// インデント文字列
    indent_str: String,
}

impl SimpleAstRenderer {
    /// 新しいレンダラーを作成
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            indent_str: "  ".to_string(),
        }
    }

    /// ASTをレンダリング
    pub fn render(&mut self, ast: &AstNode) -> String {
        match ast {
            AstNode::Const(lit) => self.render_literal(lit),
            AstNode::Var(name) => name.clone(),
            AstNode::Add(a, b) => format!("({} + {})", self.render(a), self.render(b)),
            AstNode::Mul(a, b) => format!("({} * {})", self.render(a), self.render(b)),
            AstNode::Max(a, b) => format!("max({}, {})", self.render(a), self.render(b)),
            AstNode::Rem(a, b) => format!("({} % {})", self.render(a), self.render(b)),
            AstNode::Idiv(a, b) => format!("({} / {})", self.render(a), self.render(b)),
            AstNode::Recip(a) => format!("(1 / {})", self.render(a)),
            AstNode::Sqrt(a) => format!("sqrt({})", self.render(a)),
            AstNode::Log2(a) => format!("log2({})", self.render(a)),
            AstNode::Exp2(a) => format!("exp2({})", self.render(a)),
            AstNode::Sin(a) => format!("sin({})", self.render(a)),
            AstNode::Cast(a, dtype) => format!("({:?})({})", dtype, self.render(a)),
            AstNode::Load { ptr, offset, count } => {
                if *count == 1 {
                    format!("{}[{}]", self.render(ptr), self.render(offset))
                } else {
                    format!(
                        "load{}({}[{}])",
                        count,
                        self.render(ptr),
                        self.render(offset)
                    )
                }
            }
            AstNode::Store { ptr, offset, value } => {
                format!(
                    "{}[{}] = {}",
                    self.render(ptr),
                    self.render(offset),
                    self.render(value)
                )
            }
            AstNode::Assign { var, value } => {
                format!("{} = {}", var, self.render(value))
            }
            AstNode::Block { statements, .. } => self.render_block(statements),
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => self.render_range(var, start, step, stop, body),
            AstNode::Call { name, args } => {
                let arg_strs: Vec<_> = args.iter().map(|a| self.render(a)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            AstNode::Return { value } => format!("return {}", self.render(value)),
            AstNode::Barrier => "barrier()".to_string(),
            AstNode::Wildcard(name) => format!("_{}", name),
        }
    }

    fn render_literal(&self, lit: &Literal) -> String {
        match lit {
            Literal::F32(v) => format!("{}", v),
            Literal::Isize(v) => format!("{}", v),
            Literal::Usize(v) => format!("{}", v),
        }
    }

    fn render_block(&mut self, statements: &[AstNode]) -> String {
        if statements.is_empty() {
            return "{}".to_string();
        }

        let mut result = "{\n".to_string();
        self.indent_level += 1;

        for stmt in statements {
            result.push_str(&self.indent());
            result.push_str(&self.render(stmt));
            result.push_str(";\n");
        }

        self.indent_level -= 1;
        result.push_str(&self.indent());
        result.push('}');
        result
    }

    fn render_range(
        &mut self,
        var: &str,
        start: &AstNode,
        step: &AstNode,
        stop: &AstNode,
        body: &AstNode,
    ) -> String {
        let start_str = self.render(start);
        let step_str = self.render(step);
        let stop_str = self.render(stop);

        format!(
            "for ({} = {}; {} < {}; {} += {}) {}",
            var,
            start_str,
            var,
            stop_str,
            var,
            step_str,
            self.render(body)
        )
    }

    fn indent(&self) -> String {
        self.indent_str.repeat(self.indent_level)
    }
}

impl Default for SimpleAstRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// ヘルパー関数: ASTを文字列に変換
pub fn render_ast(ast: &AstNode) -> String {
    let mut renderer = SimpleAstRenderer::new();
    renderer.render(ast)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;

    #[test]
    fn test_render_const() {
        let ast = AstNode::Const(3.14f32.into());
        assert_eq!(render_ast(&ast), "3.14");
    }

    #[test]
    fn test_render_add() {
        let ast = AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into());
        assert_eq!(render_ast(&ast), "(1 + 2)");
    }

    #[test]
    fn test_render_complex() {
        // (a + b) * 2
        let ast = (var("a") + var("b")) * AstNode::Const(2isize.into());
        assert_eq!(render_ast(&ast), "((a + b) * 2)");
    }

    #[test]
    fn test_render_sqrt() {
        let ast = sqrt(AstNode::Const(4.0f32.into()));
        assert_eq!(render_ast(&ast), "sqrt(4)");
    }
}
