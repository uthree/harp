use crate::backend::{Buffer, Compiler, Renderer};

pub struct GenericBackend<C, R, B, RendererOption, CodeRepr, CompilerOption>
where
    C: Compiler<B, CodeRepr, CompilerOption>,
    R: Renderer<CodeRepr, RendererOption>,
    B: Buffer,
{
    compiler: C,
    renderer: R,
}
