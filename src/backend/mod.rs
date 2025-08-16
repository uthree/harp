use crate::{
    ast::{AstNode, DType},
    graph::{Graph, GraphSignature},
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
pub trait Compiler<B: Buffer> {
    type CodeRepr;
    type KernelType: Kernel<B>;
    type Option;
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: Self::Option);
    fn compile(&mut self, code: &Self::CodeRepr, details: GraphSignature) -> Self::KernelType;
}
pub trait Renderer {
    type CodeRepr;
    type Option;
    fn new() -> Self;
    fn with_option(&mut self, option: Self::Option);
    fn render(&mut self, ast: AstNode) -> Self::CodeRepr;
}

pub trait Backend<B: Buffer> {
    type Option;
    fn new() -> Self;
    fn with_option(&mut self, option: Self::Option);
    fn is_available(&self) -> bool;
    fn execute(&mut self, graph: &Graph, inputs: Vec<B>) -> Vec<B>;
}
