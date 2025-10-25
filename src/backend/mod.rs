use crate::ast::AstNode;

// レンダラー。
// ASTを受け取って文字列としてレンダリングする
pub trait Renderer {
    type CodeRepr;
    fn render(&self, ast: AstNode) -> Self::CodeRepr;
}
pub trait Compiler {}
pub trait Buffer {}
pub trait Kernel {}
pub struct Query {}
