use crate::{ast::AstNode, graph::GraphSignature};

pub trait Buffer {
    fn shape(&self) -> Vec<usize>;
}

pub trait Kernel<B: Buffer> {
    /// Returns detailed information about the kernel's inputs and outputs.
    fn details(&self) -> &GraphSignature;

    /// Executes the kernel with the given buffers and shape variables.
    fn call(&mut self, buffers: Vec<B>, shape_variables: &[usize]) -> Vec<B>;
}

/// A trait for a compiler that turns a code representation into a `Kernel`.
pub trait Compiler<B: Buffer, CodeRepr = String, CompilerOption = ()> {
    type KernelType: Kernel<B>;
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: CompilerOption);
    fn compile(&mut self, code: &CodeRepr, details: GraphSignature) -> Self::KernelType;
}
pub trait Renderer<CodeRepr = String> {
    fn new() -> Self;
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}
