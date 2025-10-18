use crate::ast::AstNode;

pub trait Renderer {
    type CodeRepr;
    fn render(&mut self, ast: AstNode) -> Self::CodeRepr;
}

pub trait Compiler {
    type CodeRepr;
}

pub trait Executable {}

pub trait Buffer {}
