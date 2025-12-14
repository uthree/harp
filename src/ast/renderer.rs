//! ASTを可読性の高い文字列に変換するモジュール

use super::AstNode;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::opencl::OpenCLRenderer;

/// ASTを文字列に変換（ジェネリックレンダラー対応）
///
/// # 型パラメータ
/// * `R` - `CLikeRenderer`を実装するレンダラー
///
/// # 例
/// ```ignore
/// use harp::backend::opencl::OpenCLRenderer;
/// use harp::ast::renderer::render_ast_with;
///
/// let renderer = OpenCLRenderer::new();
/// let code = render_ast_with(&ast, &renderer);
/// ```
pub fn render_ast_with<R>(ast: &AstNode, renderer: &R) -> String
where
    R: CLikeRenderer + Clone,
{
    // AstNodeの種類によって適切なレンダリングメソッドを使用
    match ast {
        // Program全体をレンダリング
        AstNode::Program { .. } => {
            let mut mutable_renderer = renderer.clone();
            mutable_renderer.render_program_clike(ast)
        }
        // 単一のFunctionをレンダリング
        AstNode::Function { .. } => {
            let mut mutable_renderer = renderer.clone();
            mutable_renderer.render_function_node(ast)
        }
        // 文として扱うべきノード
        AstNode::Store { .. }
        | AstNode::Assign { .. }
        | AstNode::Block { .. }
        | AstNode::Range { .. }
        | AstNode::If { .. }
        | AstNode::Barrier => {
            // 文の場合はインデントなしでレンダリング
            let mut mutable_renderer = renderer.clone();
            mutable_renderer.render_statement(ast).trim().to_string()
        }
        // 式として扱うべきノード
        _ => renderer.render_expr(ast),
    }
}

/// ヘルパー関数: ASTを文字列に変換（デフォルトでOpenCLRendererを使用）
///
/// より詳細な制御が必要な場合は、`render_ast_with`を使用してください。
pub fn render_ast(ast: &AstNode) -> String {
    let renderer = OpenCLRenderer::new();
    render_ast_with(ast, &renderer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::{DType, Scope};

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_render_const() {
        let ast = AstNode::Const(3.14f32.into());
        assert_eq!(render_ast(&ast), "3.14f");
    }

    #[test]
    fn test_render_add() {
        let ast = AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into());
        assert_eq!(render_ast(&ast), "(1.0f + 2.0f)");
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
        // OpenCLRendererは generic な sqrt を使用
        assert_eq!(render_ast(&ast), "sqrt(4.0f)");
    }

    #[test]
    fn test_render_function() {
        // 簡単な関数をテスト: void foo() { return; }
        let params = vec![];
        let return_type = DType::Tuple(vec![]);
        let scope = Scope::new();
        let body = Box::new(AstNode::Block {
            statements: vec![],
            scope: Box::new(scope),
        });

        let func = AstNode::Function {
            name: Some("foo".to_string()),
            params,
            return_type,
            body,
        };

        let rendered = render_ast(&func);
        // 関数全体がレンダリングされることを確認
        assert!(rendered.contains("foo"));
        assert!(rendered.contains("{"));
        assert!(rendered.contains("}"));
    }

    #[test]
    fn test_render_program() {
        // 簡単なプログラムをテスト
        let scope = Scope::new();
        let body = Box::new(AstNode::Block {
            statements: vec![],
            scope: Box::new(scope),
        });

        let func = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body,
        };

        let program = AstNode::Program {
            functions: vec![func],
            entry_point: "main".to_string(),
        };

        let rendered = render_ast(&program);
        // Program全体がレンダリングされることを確認
        assert!(rendered.contains("main"));
        assert!(rendered.contains("Entry Point"));
    }
}
