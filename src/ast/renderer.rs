//! ASTを可読性の高い文字列に変換するモジュール

use super::AstNode;
use crate::backend::c_like::CLikeRenderer;

/// ASTを文字列に変換（ジェネリックレンダラー対応）
///
/// # 型パラメータ
/// * `R` - `CLikeRenderer`を実装するレンダラー
///
/// # 例
/// ```ignore
/// use harp::backend::opencl::OpenCLRenderer;
/// use harp::ast::{AstNode, renderer::render_ast_with};
///
/// let renderer = OpenCLRenderer::new();
/// let ast = AstNode::Program {
///     functions: vec![],
///     execution_waves: vec![],
/// };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::{DType, Scope, VarDecl};
    use crate::backend::Renderer;

    /// テスト用のダミーレンダラー（C言語風出力）
    #[derive(Clone)]
    struct TestRenderer {
        indent_level: usize,
    }

    impl TestRenderer {
        fn new() -> Self {
            Self { indent_level: 0 }
        }
    }

    impl Renderer for TestRenderer {
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

    impl CLikeRenderer for TestRenderer {
        fn indent_level(&self) -> usize {
            self.indent_level
        }
        fn indent_level_mut(&mut self) -> &mut usize {
            &mut self.indent_level
        }
        fn indent_size(&self) -> usize {
            4
        }
        fn render_dtype_backend(&self, dtype: &DType) -> String {
            match dtype {
                DType::Bool => "int".to_string(),
                DType::F32 => "float".to_string(),
                DType::Tuple(ts) if ts.is_empty() => "void".to_string(),
                DType::Tuple(ts) => format!(
                    "({})",
                    ts.iter()
                        .map(|t| self.render_dtype_backend(t))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                DType::Ptr(inner) => format!("{}*", self.render_dtype_backend(inner)),
                _ => "unknown".to_string(),
            }
        }
        fn render_barrier_backend(&self) -> String {
            "barrier();".to_string()
        }
        fn render_header(&self) -> String {
            String::new()
        }
        fn render_function_qualifier(&self, _is_kernel: bool) -> String {
            String::new()
        }
        fn render_param_attribute(&self, _param: &VarDecl, _is_kernel: bool) -> String {
            String::new()
        }
        fn render_thread_var_declarations(&self, _params: &[VarDecl], _indent: &str) -> String {
            String::new()
        }
        fn render_math_func(&self, name: &str, args: &[String]) -> String {
            format!("{}({})", name, args.join(", "))
        }
        fn render_atomic_add(
            &self,
            ptr: &str,
            offset: &str,
            value: &str,
            _dtype: &DType,
        ) -> String {
            format!("atomic_add(&{}[{}], {})", ptr, offset, value)
        }
        fn render_atomic_max(
            &self,
            ptr: &str,
            offset: &str,
            value: &str,
            _dtype: &DType,
        ) -> String {
            format!("atomic_max(&{}[{}], {})", ptr, offset, value)
        }
    }

    fn render_ast(ast: &super::AstNode) -> String {
        let renderer = TestRenderer::new();
        render_ast_with(ast, &renderer)
    }

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
        let ast = (var("a") + var("b")) * AstNode::Const(2i64.into());
        assert_eq!(render_ast(&ast), "((a + b) * 2)");
    }

    #[test]
    fn test_render_sqrt() {
        let ast = sqrt(AstNode::Const(4.0f32.into()));
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
            execution_waves: vec![],
        };

        let rendered = render_ast(&program);
        // Program全体がレンダリングされることを確認
        assert!(rendered.contains("main"));
    }
}
