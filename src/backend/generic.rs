use crate::{
    ast::AstNode,
    backend::{Buffer, Compiler, Renderer},
    graph::GraphSignature,
};
use std::marker::PhantomData;

pub struct GenericBackend<C, R, B>
where
    R: Renderer,
    C: Compiler<B, R::CodeRepr>,
    B: Buffer,
{
    compiler: C,
    renderer: R,
    // To store options
    #[allow(dead_code)]
    compiler_option: Option<C::Option>,
    #[allow(dead_code)]
    renderer_option: Option<R::Option>,
    _phantom: PhantomData<(B, R::CodeRepr)>,
}

impl<C, R, B> GenericBackend<C, R, B>
where
    R: Renderer,
    C: Compiler<B, R::CodeRepr>,
    B: Buffer,
{
    pub fn new() -> Self {
        Self {
            compiler: C::new(),
            renderer: R::new(),
            compiler_option: None,
            renderer_option: None,
            _phantom: PhantomData,
        }
    }

    pub fn with_options(
        &mut self,
        compiler_option: C::Option,
        renderer_option: R::Option,
    ) -> &mut Self {
        self.compiler.with_option(compiler_option);
        self.renderer.with_option(renderer_option);
        self
    }

    pub fn compile(&mut self, ast: AstNode, details: GraphSignature) -> C::KernelType {
        let code = self.renderer.render(ast);
        self.compiler.compile(&code, details)
    }
}

impl<C, R, B> Default for GenericBackend<C, R, B>
where
    R: Renderer,
    C: Compiler<B, R::CodeRepr>,
    B: Buffer,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::DType,
        backend::{Buffer, Kernel, c::CBackend},
        graph::{Graph, TensorSignature, shape::expr::Expr as ShapeExpr},
    };

    #[test]
    fn test_generic_backend_with_c() {
        // 1. Build a graph
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(1)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b;
        graph.outputs.push(c);
        graph.signature.outputs.push(TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        });

        // 2. Lower the graph to AST
        let ast = crate::lowerer::lower_graph(&graph);

        // 3. Use GenericBackend (via CBackend) to compile
        let mut backend = CBackend::new();
        let mut kernel = backend.compile(ast, graph.signature);

        // 4. Prepare buffers and run the kernel
        let a_data = vec![1.0f32];
        let b_data = vec![2.0f32];
        let shape_usize = vec![1];

        let a_buffer = crate::backend::c::CBuffer::from_slice(&a_data, &shape_usize, dtype.clone());
        let b_buffer = crate::backend::c::CBuffer::from_slice(&b_data, &shape_usize, dtype.clone());
        let out_buffer = crate::backend::c::CBuffer::allocate(dtype, shape_usize);

        let buffers = vec![out_buffer, a_buffer, b_buffer];
        let result_buffers = kernel.call(buffers, &[]);

        // 5. Check the result
        let result_data = result_buffers[0].to_vec::<f32>();
        assert_eq!(result_data, vec![3.0f32]);
    }
}
