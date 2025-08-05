use crate::ast::{AstNode, DType};

pub mod c;

pub trait Device<Var: Buffer> {
    fn allocate(&mut self, dtype: DType, size: usize) -> Var;
    fn free(&mut self, var: Var);
    fn is_available(&self) -> bool;
}

pub trait Compiler<Var: Buffer, CodeRepr = String, CompilerOption = ()> {
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: CompilerOption);
    fn compile(&mut self, code: CodeRepr) -> impl Kernel<Var>;
}

pub trait Renderer<CodeRepr = String> {
    fn new() -> Self;
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}
pub trait Kernel<Var: Buffer> {
    fn call(&self, buffers: Vec<Var>) -> Vec<Var>;
}

pub trait Buffer {}
