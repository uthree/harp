use crate::opt::heuristic;
use crate::{
    backend::{Backend, Buffer, Compiler, Kernel, Renderer},
    graph::Graph,
    graph::lowerer::Lowerer,
    graph::lowerer::orchestrator::LoweringOrchestrator,
    opt::ast::{AlgebraicSimplification, DeterministicAstOptimizer},
    opt::graph::ElementwiseFusion,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Creates a stable key for an AstNode, ignoring the node's ID.
fn ast_node_key(node: &crate::ast::AstNode) -> String {
    let mut key = format!("op:{:?},dtype:{:?},srcs:[", node.op, node.dtype);
    for src in &node.src {
        key.push_str(&ast_node_key(src));
        key.push(',');
    }
    key.push(']');
    key
}

pub struct GenericBackendConfig {
    pub heuristic_optimization_threshold: usize,
}

impl Default for GenericBackendConfig {
    fn default() -> Self {
        Self {
            heuristic_optimization_threshold: 10, // Default threshold
        }
    }
}

pub struct GenericBackend<C, R, B, CodeRepr, CompilerOption>
where
    B: Buffer,
    C: Compiler<B, CodeRepr, CompilerOption>,
    R: Renderer<CodeRepr>,
{
    compiler: Mutex<C>,
    renderer: Mutex<R>,
    cache: Mutex<HashMap<String, Arc<Mutex<C::KernelType>>>>,
    call_counts: Mutex<HashMap<String, usize>>,
    config: GenericBackendConfig,
    graph_optimizer: ElementwiseFusion,
    pub compile_count: Mutex<usize>, // For testing purposes
}

impl<C, R, B, CodeRepr, CompilerOption> GenericBackend<C, R, B, CodeRepr, CompilerOption>
where
    B: Buffer,
    C: Compiler<B, CodeRepr, CompilerOption> + Send + Sync,
    R: Renderer<CodeRepr> + Send + Sync,
    CodeRepr: Send + Sync,
    CompilerOption: Send + Sync,
    C::KernelType: Send + Sync,
{
    pub fn with_config(config: GenericBackendConfig) -> Self {
        Self {
            compiler: Mutex::new(C::new()),
            renderer: Mutex::new(R::new()),
            cache: Mutex::new(HashMap::new()),
            call_counts: Mutex::new(HashMap::new()),
            config,
            graph_optimizer: ElementwiseFusion::new(),
            compile_count: Mutex::new(0),
        }
    }

    fn compile_kernel(
        &self,
        graph: &Graph,
        use_heuristic: bool,
    ) -> (C::KernelType, crate::backend::KernelDetails) {
        let mut lowerer = Lowerer::new(graph);
        let (mut ast, details) = lowerer.lower();

        if use_heuristic {
            log::debug!("Applying heuristic AST optimization...");
            let suggester = AlgebraicSimplification::new();
            let cost_estimator = heuristic::HandcodedCostEstimator;
            let optimizer = heuristic::BeamSearchAstOptimizer::new(suggester, cost_estimator, 2)
                .with_max_steps(3);
            ast = optimizer.optimize(ast);
        }

        let code = self.renderer.lock().unwrap().render(ast);
        let kernel = self
            .compiler
            .lock()
            .unwrap()
            .compile(&code, details.clone());
        (kernel, details)
    }
}

impl<C, R, B, CodeRepr, CompilerOption> Backend<B>
    for GenericBackend<C, R, B, CodeRepr, CompilerOption>
where
    B: Buffer,
    C: Compiler<B, CodeRepr, CompilerOption> + Send + Sync,
    R: Renderer<CodeRepr> + Send + Sync,
    CodeRepr: Send + Sync,
    CompilerOption: Send + Sync,
    C::KernelType: Send + Sync,
{
    fn new() -> Self {
        Self::with_config(GenericBackendConfig::default())
    }

    fn is_available(&self) -> bool {
        self.compiler.lock().unwrap().is_available()
    }

    fn execute(&self, graph: &Graph, inputs: Vec<B>, shape_variables: Vec<usize>) -> Vec<B> {
        // Always apply deterministic graph optimizations first.
        let optimized_graph = self.graph_optimizer.optimize(graph);
        let graph_to_use = if *optimized_graph.nodes.borrow() == *graph.nodes.borrow() {
            graph
        } else {
            &optimized_graph
        };

        let graph_key = {
            let mut key = String::new();
            for node in graph_to_use.nodes.borrow().iter() {
                let op_key = match &node.op {
                    crate::graph::GraphOp::FusedElementwise(ast) => {
                        format!("FusedElementwise({})", ast_node_key(ast))
                    }
                    other => format!("{:?}", other),
                };
                key.push_str(&format!(
                    "op:{},dtype:{:?},shape:{:?},src:{:?};",
                    op_key, node.dtype, node.shape, node.src
                ));
            }
            key.push_str(&format!("outputs:{:?}", graph_to_use.outputs.borrow()));
            key
        };

        let mut call_counts = self.call_counts.lock().unwrap();
        let count = call_counts.entry(graph_key.clone()).or_insert(0);
        *count += 1;
        let trigger_heuristic = *count == self.config.heuristic_optimization_threshold;

        let kernel = {
            let mut cache = self.cache.lock().unwrap();
            if let Some(kernel) = cache.get(&graph_key) {
                if !trigger_heuristic {
                    log::debug!("Backend cache hit");
                    kernel.clone()
                } else {
                    // Time to apply heuristic optimization and re-cache.
                    log::debug!("Applying heuristic optimization and recompiling...");
                    let (new_kernel, _) = self.compile_kernel(graph_to_use, true);
                    let arc_kernel = Arc::new(Mutex::new(new_kernel));
                    *self.compile_count.lock().unwrap() += 1;
                    // Overwrite the existing entry with the heuristically optimized kernel.
                    cache.insert(graph_key.clone(), arc_kernel.clone());
                    arc_kernel
                }
            } else {
                log::debug!("Backend cache miss, compiling with deterministic opts...");
                let (new_kernel, _) = self.compile_kernel(graph_to_use, false);
                let arc_kernel = Arc::new(Mutex::new(new_kernel));
                *self.compile_count.lock().unwrap() += 1;
                cache.insert(graph_key.clone(), arc_kernel.clone());
                arc_kernel
            }
        };

        let num_inputs = inputs.len();
        let mut all_buffers = inputs;
        let mut kernel_locked = kernel.lock().unwrap();

        let shape_vars_map: std::collections::HashMap<String, i64> = kernel_locked
            .details()
            .shape_variables
            .iter()
            .cloned()
            .zip(shape_variables.iter().map(|&v| v as i64))
            .collect();

        for buffer_info in kernel_locked.details().buffers.iter().skip(num_inputs) {
            let shape = buffer_info
                .shape
                .iter()
                .map(|expr| expr.evaluate(&shape_vars_map) as usize)
                .collect();
            all_buffers.push(B::allocate(buffer_info.dtype.clone(), shape));
        }

        let result_buffers = kernel_locked.call(all_buffers, &shape_variables);
        result_buffers.into_iter().skip(num_inputs).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::DType,
        backend::{Backend, c::*},
    };

    #[test]
    fn test_optimization_strategy() {
        crate::init_logger();
        // Configure a backend with a low optimization threshold for testing.
        let config = GenericBackendConfig {
            heuristic_optimization_threshold: 3,
        };
        let backend = CBackend::with_config(config);

        // Create a simple graph that can be optimized.
        let graph = Graph::new();
        let input = graph.input(DType::F32, vec![1.into()]);
        let neg_node = -input;
        let sin_node = neg_node.sin();
        let _ = sin_node.exp2().as_output();

        // The first call should compile the deterministically optimized kernel.
        assert_eq!(*backend.compile_count.lock().unwrap(), 0);
        let _ = backend.execute(&graph, vec![], vec![]);
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            1,
            "First call should compile"
        );

        // Subsequent calls before the threshold should use the cache.
        let _ = backend.execute(&graph, vec![], vec![]);
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            1,
            "Second call should hit cache"
        );

        // The call at the threshold should trigger heuristic optimization and re-compilation.
        let _ = backend.execute(&graph, vec![], vec![]);
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            2,
            "Third call (at threshold) should recompile with heuristics"
        );

        // Calls after optimization should use the new heuristically optimized kernel from cache.
        let _ = backend.execute(&graph, vec![], vec![]);
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            2,
            "Fourth call should hit heuristic cache"
        );
        let _ = backend.execute(&graph, vec![], vec![]);
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            2,
            "Fifth call should hit heuristic cache"
        );
    }
}
