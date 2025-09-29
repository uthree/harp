use crate::ast::DType;
use crate::ast::Program;
use crate::graph::Graph;
use crate::graph::GraphSignature;
pub mod c;
pub mod generic;
pub trait Buffer {
    // get buffer size
    fn shape(&self) -> Vec<usize>;
    fn allocate(dtype: DType, shape: Vec<usize>) -> Self; // メモリーを確保する
                                                          // WARNING: dropされたときにメモリが解放されるようにしてください。
}

pub trait Kernel {
    type Buffer: Buffer;
    /// Returns detailed information about the kernel's inputs and outputs.
    fn signature(&self) -> &GraphSignature;

    /// Executes the kernel with the given buffers and shape variables.
    fn call(&mut self, buffers: Vec<Self::Buffer>, shape_variables: &[usize]) -> Vec<Self::Buffer>;
}

/// A trait for a compiler that turns a code representation into a `Kernel`.
pub trait Compiler {
    type CodeRepr;
    type Buffer: Buffer;
    type KernelType: Kernel<Buffer = Self::Buffer>;
    type Option;
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: Self::Option) {} // default implementation is "do nothing".
    fn compile(&mut self, code: &Self::CodeRepr, details: GraphSignature) -> Self::KernelType;
}
pub trait Renderer {
    type CodeRepr;
    type Option;
    fn new() -> Self;
    fn with_option(&mut self, option: Self::Option) {} // default implementation is "do nothing".
    fn render(&mut self, program: Program) -> Self::CodeRepr;
}

pub trait Backend {
    type Buffer: Buffer;
    type Option;
    type Compiler: Compiler;
    type Renderer: Renderer<CodeRepr = <Self::Compiler as Compiler>::CodeRepr>;
    fn new() -> Self;
    fn with_option(&mut self, option: Self::Option);
    fn is_available(&self) -> bool;
    fn execute(&mut self, graph: &Graph, inputs: Vec<Self::Buffer>) -> Vec<Self::Buffer>;
}
