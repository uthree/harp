use crate::ast::AstNode;

pub trait Renderer {
    type CodeRepr;
    fn render(&self, ast: AstNode) -> Self::CodeRepr;
}
pub trait Compiler {}
pub trait Buffer {}
pub trait Kernel {}
pub struct Query {}
