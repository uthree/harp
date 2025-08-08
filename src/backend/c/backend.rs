use super::{CBuffer, CCompiler, CKernel, CRenderer};
use crate::{
    backend::{Backend, Buffer, Compiler, Kernel, Renderer},
    graph::{Graph, lowerer::Lowerer},
};
use rustc_hash::FxHashMap;
use std::collections::{HashMap, hash_map::Entry};

pub struct CBackend {
    graph_cache: FxHashMap<String, CKernel>,
}

impl Backend<CBuffer> for CBackend {
    fn new() -> Self {
        CBackend {
            graph_cache: FxHashMap::default(),
        }
    }

    fn is_available(&self) -> bool {
        CCompiler::new().is_available()
    }
    fn call(
        &mut self,
        graph: Graph,
        buffers: Vec<CBuffer>,
        shape_variables: Vec<usize>,
    ) -> Vec<CBuffer> {
        log::info!("CBackend::call with graph: {:?}", graph);
        let key = format!("{graph:?}");
        // 1. Get the kernel from cache or compile it.
        let kernel = match self.graph_cache.entry(key) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                // Lower the graph to AST
                let mut lowerer = Lowerer::new(&graph);
                let (ast, details) = lowerer.lower();

                // Render the AST to C code
                let mut renderer = CRenderer::new();
                let code = renderer.render(ast);

                // Compile the C code
                let mut compiler = CCompiler::new();
                let kernel = compiler.compile(&code, details);

                entry.insert(kernel.clone());
                kernel
            }
        };

        // 2. Prepare buffers for the kernel call.
        let details = kernel.details();
        let num_inputs = graph.inputs.borrow().len();
        let num_outputs = graph.outputs.borrow().len();
        let num_total_buffers = details.buffers.len();

        assert_eq!(
            buffers.len(),
            num_inputs,
            "Incorrect number of input buffers provided"
        );

        let mut all_buffers: Vec<Option<CBuffer>> = (0..num_total_buffers).map(|_| None).collect();
        for (i, buffer) in buffers.into_iter().enumerate() {
            all_buffers[i] = Some(buffer);
        }

        let shape_vars_map: HashMap<String, i64> = details
            .shape_variables
            .iter()
            .cloned()
            .zip(shape_variables.iter().map(|&v| v as i64))
            .collect();

        for i in num_inputs..num_total_buffers {
            if all_buffers[i].is_none() {
                let buffer_info = &details.buffers[i];
                let shape: Vec<usize> = buffer_info
                    .shape
                    .iter()
                    .map(|expr| expr.evaluate(&shape_vars_map) as usize)
                    .collect();
                let dtype = buffer_info.dtype.clone();
                all_buffers[i] = Some(CBuffer::allocate(dtype, shape));
            }
        }

        let final_buffers = all_buffers.into_iter().map(|b| b.unwrap()).collect();

        // 3. Call the kernel.
        let mut result_buffers = kernel.call(final_buffers, &shape_variables);

        // 4. Extract and return the output buffer(s).
        assert_eq!(result_buffers.len(), num_total_buffers);
        result_buffers.drain(num_inputs..num_inputs + num_outputs).collect()
    }
}