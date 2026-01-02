use crate::ast;

pub trait Renderer {
    type CodeRepr;
    fn render(&self, ast: ast::Node) -> Self::CodeRepr;
}

pub trait Compiler {
    type CodeRepr;
    type Kernel;

    fn compile(&self, code: Self::CodeRepr) -> Self::Kernel;
}

pub trait Buffer {}

pub trait Kernel {}

pub struct Query {}
