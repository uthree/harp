use super::{CBuffer, CCompiler, CKernel, CRenderer};
use crate::{
    backend::{Backend, Compiler, Kernel, Renderer},
    graph::Graph,
    graph::lowerer::Lowerer,
    graph::lowerer::orchestrator::LoweringOrchestrator,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct CBackend {
    compiler: RefCell<CCompiler>,
    renderer: RefCell<CRenderer>,
    cache: RefCell<HashMap<String, Arc<Mutex<CKernel>>>>,
    pub compile_count: RefCell<usize>, // For testing purposes
}

impl Backend<CBuffer> for CBackend {
    fn new() -> Self {
        CBackend {
            compiler: RefCell::new(CCompiler::new()),
            renderer: RefCell::new(CRenderer::new()),
            cache: RefCell::new(HashMap::new()),
            compile_count: RefCell::new(0),
        }
    }

    fn is_available(&self) -> bool {
        self.compiler.borrow().is_available()
    }

    fn run(&mut self, graph: &Graph) -> Vec<CBuffer> {
        self.execute(graph, vec![], vec![])
    }

    fn execute(
        &mut self,
        graph: &Graph,
        inputs: Vec<CBuffer>,
        shape_variables: Vec<usize>,
    ) -> Vec<CBuffer> {
        // 1. Generate a unique key from the graph structure.
        // This key represents the computation itself.
        let graph_key = {
            let nodes_repr = format!("{:?}", graph.nodes.borrow());
            let outputs_repr = format!("{:?}", graph.outputs.borrow());
            format!("nodes:{};outputs:{}", nodes_repr, outputs_repr)
        };

        // 2. Check cache using the graph key.
        let mut cache = self.cache.borrow_mut();
        let kernel = if let Some(kernel) = cache.get(&graph_key) {
            log::debug!("CBackend cache hit");
            kernel.clone()
        } else {
            log::debug!("CBackend cache miss, compiling...");
            // 3. If miss, lower the graph to get the AST and kernel details.
            let mut lowerer = Lowerer::new(graph);
            let (ast, details) = lowerer.lower();

            // 4. Render the AST to C code.
            let code = self.renderer.get_mut().render(ast);

            // 5. Compile the C code into a kernel.
            let new_kernel = self.compiler.get_mut().compile(&code, details);
            let arc_kernel = Arc::new(Mutex::new(new_kernel));
            *self.compile_count.borrow_mut() += 1;

            // 6. Store the new kernel in the cache with the graph key.
            cache.insert(graph_key, arc_kernel.clone());
            arc_kernel
        };

        // 7. Prepare buffers for the kernel call.
        let num_inputs = inputs.len();
        let mut all_buffers = inputs;

        // Allocate output buffers.
        let mut kernel_locked = kernel.lock().unwrap();
        let shape_vars_map: std::collections::HashMap<String, i64> = kernel_locked
            .details
            .shape_variables
            .iter()
            .cloned()
            .zip(shape_variables.iter().map(|&v| v as i64))
            .collect();

        for buffer_info in kernel_locked.details.buffers.iter().skip(num_inputs) {
            let shape = buffer_info
                .shape
                .iter()
                .map(|expr| expr.evaluate(&shape_vars_map) as usize)
                .collect();
            all_buffers.push(CBuffer::allocate(buffer_info.dtype.clone(), shape));
        }

        // 8. Execute the kernel.
        let result_buffers = kernel_locked.call(all_buffers, &shape_variables);

        // 9. Return only the output buffers.
        result_buffers.into_iter().skip(num_inputs).collect()
    }
}
