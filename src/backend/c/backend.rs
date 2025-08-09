use super::{CBuffer, CCompiler, CRenderer};
use crate::{
    backend::{Backend, Buffer, Compiler, Kernel, Renderer},
    graph::Graph,
    graph::lowerer::orchestrator::LoweringOrchestrator,
};

pub struct CBackend {
    compiler: CCompiler,
    renderer: CRenderer,
}

impl Backend for CBackend {
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
        inputs: Vec<Box<dyn Buffer>>,
        shape_variables: Vec<usize>,
    ) -> Vec<Box<dyn Buffer>> {
        // 1. Lower the graph to get the AST and kernel details.
        let (ast, details) = crate::graph::lowerer::Lowerer::new(&graph).lower();

        // 2. Render the AST to C code.
        let code = self.renderer.render(ast);

        // 3. Compile the C code into a kernel.
        let mut kernel = self.compiler.compile(&code, details);

        // 4. Prepare buffers for the kernel call.
        let num_inputs = inputs.len();

        // Downcast the input buffers from `Box<dyn Buffer>` to `CBuffer`.
        // This is necessary because the C backend's kernel expects concrete `CBuffer` types.
        // A clone is performed here. For performance-critical applications,
        // this might need optimization to avoid cloning.
        let mut all_buffers: Vec<CBuffer> = inputs
            .into_iter()
            .map(|buf| {
                buf.as_any()
                    .downcast_ref::<CBuffer>()
                    .expect("CBackend requires CBuffer inputs")
                    .clone()
            })
            .collect();

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

        // 6. Return only the output buffers, upcasting them to `Box<dyn Buffer>`.
        result_buffers
            .into_iter()
            .skip(num_inputs)
            .map(|b| Box::new(b) as Box<dyn Buffer>)
            .collect()
    }
}
