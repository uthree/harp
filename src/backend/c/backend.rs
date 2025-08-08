use super::{CBuffer, CCompiler, CRenderer};
use crate::{
    backend::{Backend, Buffer, Compiler, Kernel, Renderer},
    graph::Graph,
};

pub struct CBackend {
    compiler: CCompiler,
    renderer: CRenderer,
}

impl Backend<CBuffer> for CBackend {
    fn new() -> Self {
        CBackend {
            compiler: CCompiler::new(),
            renderer: CRenderer::new(),
        }
    }

    fn is_available(&self) -> bool {
        self.compiler.is_available()
    }

    fn call(
        &mut self,
        graph: Graph,
        inputs: Vec<CBuffer>,
        shape_variables: Vec<usize>,
    ) -> Vec<CBuffer> {
        // 1. Lower the graph to get the AST and kernel details.
        let (ast, details) = crate::graph::lowerer::Lowerer::new(&graph).lower();

        // 2. Render the AST to C code.
        let code = self.renderer.render(ast);

        // 3. Compile the C code into a kernel.
        let kernel = self.compiler.compile(&code, details);

        // 4. Prepare buffers for the kernel call.
        let num_inputs = inputs.len();
        let mut all_buffers = inputs;

        // Allocate output buffers.
        let shape_vars_map: std::collections::HashMap<String, i64> = kernel
            .details
            .shape_variables
            .iter()
            .cloned()
            .zip(shape_variables.iter().map(|&v| v as i64))
            .collect();

        for buffer_info in kernel.details.buffers.iter().skip(num_inputs) {
            let shape = buffer_info
                .shape
                .iter()
                .map(|expr| expr.evaluate(&shape_vars_map) as usize)
                .collect();
            all_buffers.push(CBuffer::allocate(buffer_info.dtype.clone(), shape));
        }

        // 5. Execute the kernel.
        let result_buffers = kernel.call(all_buffers, &shape_variables);

        // 6. Return only the output buffers.
        result_buffers.into_iter().skip(num_inputs).collect()
    }
}