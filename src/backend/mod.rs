use crate::{
    ast::{AstNode, DType},
    tensor::Tensor,
};

pub trait Device<Buffer> {
    fn allocate(&mut self, dtype: DType, size: usize) -> Buffer;
    fn free(&mut self, buffer: Buffer);
    fn is_available(&self) -> bool;
}

pub trait Compiler<Kernel, CodeRepr = String, CompilerOption = ()> {
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: CompilerOption);
    fn compile(&mut self, code: CodeRepr) -> Kernel;
}

pub trait Renderer<CodeRepr = String> {
    fn new() -> Self;
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}
pub trait Kernel<Buffer> {
    fn call(&self, buffers: Vec<Buffer>) -> Vec<Buffer>;
}
