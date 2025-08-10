use super::{CBuffer, CCompiler, CRenderer};
use crate::{
    backend::{Backend, Compiler, Kernel, Renderer},
    graph::Graph,
    graph::lowerer::Lowerer,
    graph::lowerer::orchestrator::LoweringOrchestrator,
};
use std::cell::RefCell;

pub struct CBackend {
    compiler: RefCell<CCompiler>,
    renderer: RefCell<CRenderer>,
}

impl Backend<CBuffer> for CBackend {
    fn new() -> Self {
        CBackend {
            compiler: RefCell::new(CCompiler::new()),
            renderer: RefCell::new(CRenderer::new()),
        }
    }

    fn is_available(&self) -> bool {
        self.compiler.borrow().is_available()
    }

    fn run(&self, graph: &Graph) -> CBuffer {
        let mut lowerer = Lowerer::new(graph);
        let (ast, details) = lowerer.lower();

        let code = self.renderer.borrow_mut().render(ast);
        let mut kernel = self.compiler.borrow_mut().compile(&code, details);

        let mut buffers = vec![];
        let shape_vars_map: std::collections::HashMap<String, i64> = kernel
            .details()
            .shape_variables
            .iter()
            .cloned()
            .zip(Vec::<usize>::new().iter().map(|&v| v as i64))
            .collect();

        for buffer_info in kernel.details().buffers.iter() {
            let shape = buffer_info
                .shape
                .iter()
                .map(|expr| expr.evaluate(&shape_vars_map) as usize)
                .collect();
            buffers.push(CBuffer::allocate(buffer_info.dtype.clone(), shape));
        }

        let result_buffers = kernel.call(buffers, &vec![]);
        result_buffers.into_iter().last().unwrap()
    }

    fn call(
        &mut self,
        graph: Graph,
        inputs: Vec<CBuffer>,
        shape_variables: Vec<usize>,
    ) -> Vec<CBuffer> {
        // 1. Lower the graph to get the AST and kernel details.
        let mut lowerer = Lowerer::new(&graph);
        let (ast, details) = lowerer.lower();

        // 2. Render the AST to C code.
        let code = self.renderer.get_mut().render(ast);

        // 3. Compile the C code into a kernel.
        let mut kernel = self.compiler.get_mut().compile(&code, details);

        // 4. Prepare buffers for the kernel call.
        let num_inputs = inputs.len();
        let mut all_buffers = inputs;

        // Allocate output buffers.
        let shape_vars_map: std::collections::HashMap<String, i64> = kernel
            .details()
            .shape_variables
            .iter()
            .cloned()
            .zip(shape_variables.iter().map(|&v| v as i64))
            .collect();

        for buffer_info in kernel.details().buffers.iter().skip(num_inputs) {
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
