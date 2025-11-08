//! ASTを可読性の高い文字列に変換するモジュール

use super::AstNode;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::openmp::CRenderer;

/// ヘルパー関数: ASTを文字列に変換（backendのCRendererを使用）
pub fn render_ast(ast: &AstNode) -> String {
    let renderer = CRenderer::new();

    // AstNodeの種類によって適切なレンダリングメソッドを使用
    match ast {
        // 文として扱うべきノード
        AstNode::Store { .. }
        | AstNode::Assign { .. }
        | AstNode::Block { .. }
        | AstNode::Range { .. }
        | AstNode::Barrier => {
            // 文の場合はインデントなしでレンダリング
            let mut mutable_renderer = renderer.clone();
            mutable_renderer.render_statement(ast).trim().to_string()
        }
        // 式として扱うべきノード
        _ => renderer.render_expr(ast),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;

    #[test]
    fn test_render_const() {
        let ast = AstNode::Const(3.14f32.into());
        assert_eq!(render_ast(&ast), "3.14f");
    }

    #[test]
    fn test_render_add() {
        let ast = AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into());
        assert_eq!(render_ast(&ast), "(1f + 2f)");
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
        assert_eq!(render_ast(&ast), "sqrtf(4f)");
    }
}
