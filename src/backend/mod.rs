use crate::{
    ast::{AstNode, DType},
    graph::GraphSignature,
};

pub mod c;
pub mod generic;

pub trait Buffer {
    // get buffer size
    fn shape(&self) -> Vec<usize>;
    fn allocate(dtype: DType, shape: Vec<usize>) -> Self; // メモリーを確保する
    // WARNING: dropされたときにメモリが解放されるようにしてください。
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
    fn with_option(&mut self, option: CompilerOption) {}
    fn compile(&mut self, code: &CodeRepr, details: GraphSignature) -> Self::KernelType;
}
pub trait Renderer<CodeRepr = String, RendererOption = ()> {
    fn new() -> Self;
    fn with_option(&mut self, option: RendererOption) {}
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}

pub trait Backend<B: Buffer, BackendOption = ()> {
    fn new() -> Self;
    fn with_option(&mut self, option: BackendOption);
    fn is_available(&self) -> bool;
    fn execute(&mut self, inputs: Vec<B>) -> Vec<B>;
}
