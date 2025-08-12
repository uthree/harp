use crate::opt::heuristic;
use crate::{
    backend::{Backend, Buffer, Compiler, Kernel, KernelDetails, Renderer},
    graph::Graph,
    graph::lowerer::Lowerer,
    graph::lowerer::orchestrator::LoweringOrchestrator,
    opt::DeterministicGraphOptimizer,
    opt::ast::{AlgebraicSimplification, DeterministicAstOptimizer},
    opt::graph::ElementwiseFusion,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

/// Recursively builds a key for a node based on its logical structure, ignoring NodeId.
fn build_logical_key(
    node_id: crate::graph::NodeId,
    graph: &Graph,
    memo: &mut HashMap<crate::graph::NodeId, String>,
) -> String {
    if let Some(key) = memo.get(&node_id) {
        return key.clone();
    }

    let node = &graph.nodes.borrow()[node_id.0];
    let op_key = match &node.op {
        crate::graph::GraphOp::FusedElementwise(ast) => {
            format!("FusedElementwise({})", ast_node_key(ast))
        }
        other => format!("{:?}", other),
    };
    let mut key = format!(
        "op:{},dtype:{:?},shape:{:?},srcs:[",
        op_key, node.dtype, node.shape
    );

    for src_id in &node.src {
        key.push_str(&build_logical_key(*src_id, graph, memo));
        key.push(',');
    }
    key.push(']');

    memo.insert(node_id, key.clone());
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
    C: Compiler<B, CodeRepr, CompilerOption> + Send,
    R: Renderer<CodeRepr> + Send,
    CodeRepr: Send,
    CompilerOption: Send,
    C::KernelType: Send,
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
        // Apply deterministic graph optimizations (e.g., element-wise fusion).
        // This ensures that the kernel is always compiled from an optimized graph representation.
        // Note: This might be redundant if the graph is already optimized before calling,
        // but it makes this function more self-contained.
        let optimized_graph = self.graph_optimizer.optimize(graph);
        let graph_to_use = if *optimized_graph.nodes.borrow() == *graph.nodes.borrow() {
            graph
        } else {
            &optimized_graph
        };

        let mut lowerer = Lowerer::new(graph_to_use);
        let (mut ast, details) = lowerer.lower();

        if use_heuristic {
            log::debug!("Applying heuristic AST optimization...");
            let suggester = AlgebraicSimplification::new();
            let cost_estimator = heuristic::ExecutionTimeCostEstimator::new();
            let optimizer = heuristic::BeamSearchAstOptimizer::new(suggester, cost_estimator)
                .with_beam_width(4)
                .with_max_steps(1000);
            ast = optimizer.optimize(ast, &details);
        }

        let code = self.renderer.lock().unwrap().render(ast);
        let kernel = self
            .compiler
            .lock()
            .unwrap()
            .compile(&code, details.clone());
        (kernel, details)
    }

    pub fn measure_ast_execution_time(
        &self,
        ast: &crate::ast::AstNode,
        details: &KernelDetails,
    ) -> f32 {
        let code = self.renderer.lock().unwrap().render(ast.clone());

        // コンパイル
        let mut kernel = self
            .compiler
            .lock()
            .unwrap()
            .compile(&code, details.clone());

        // ダミーバッファの準備
        let (dummy_inputs, dummy_shape_vars) = self.prepare_dummy_buffers(details);

        // 時間計測
        let start = Instant::now();
        let _ = kernel.call(dummy_inputs, &dummy_shape_vars);
        let duration = start.elapsed();

        duration.as_secs_f32()
    }

    pub fn prepare_dummy_buffers(&self, details: &KernelDetails) -> (Vec<B>, Vec<usize>) {
        let shape_vars_map: std::collections::HashMap<String, i64> = details
            .shape_variables
            .iter()
            .map(|v| (v.clone(), 0i64)) // Use 0 for all shape vars
            .collect();

        let inputs = details
            .buffers
            .iter()
            .map(|info| {
                let shape = info
                    .shape
                    .iter()
                    .map(|expr| expr.evaluate(&shape_vars_map) as usize)
                    .collect();
                B::allocate(info.dtype.clone(), shape)
            })
            .collect();
        let shape_vars = vec![0; details.shape_variables.len()];
        (inputs, shape_vars)
    }
}

impl<C, R, B, CodeRepr, CompilerOption> Backend<B>
    for GenericBackend<C, R, B, CodeRepr, CompilerOption>
where
    B: Buffer,
    C: Compiler<B, CodeRepr, CompilerOption> + Send,
    R: Renderer<CodeRepr> + Send,
    CodeRepr: Send,
    CompilerOption: Send,
    C::KernelType: Send,
{
    fn new() -> Self {
        Self::with_config(GenericBackendConfig::default())
    }

    fn is_available(&self) -> bool {
        self.compiler.lock().unwrap().is_available()
    }

    fn execute(
        &self,
        graph: &Graph,
        inputs: Vec<B>,
        shape_variables: Vec<usize>,
    ) -> HashMap<crate::graph::NodeId, B> {
        // Always apply deterministic graph optimizations first.
        let optimized_graph = self.graph_optimizer.optimize(graph);
        let graph_to_use = if *optimized_graph.nodes.borrow() == *graph.nodes.borrow() {
            graph
        } else {
            &optimized_graph
        };

        let graph_key = {
            let mut memo = HashMap::new();
            let mut key = String::new();
            for output_id in graph_to_use.outputs.borrow().iter() {
                key.push_str(&build_logical_key(*output_id, graph_to_use, &mut memo));
            }
            key
        };

        let mut call_counts = self.call_counts.lock().unwrap();
        let count = call_counts.entry(graph_key.clone()).or_insert(0);
        *count += 1;
        let trigger_heuristic = *count == self.config.heuristic_optimization_threshold;

        let (kernel, details) = {
            let mut cache = self.cache.lock().unwrap();
            if let Some(kernel) = cache.get(&graph_key) {
                if !trigger_heuristic {
                    log::debug!("Backend cache hit");
                    // Need to get details from the kernel
                    let details = kernel.lock().unwrap().details().clone();
                    (kernel.clone(), details)
                } else {
                    // Time to apply heuristic optimization and re-cache.
                    log::debug!("Applying heuristic optimization and recompiling...");
                    let (new_kernel, details) = self.compile_kernel(graph_to_use, true);
                    let arc_kernel = Arc::new(Mutex::new(new_kernel));
                    *self.compile_count.lock().unwrap() += 1;
                    // Overwrite the existing entry with the heuristically optimized kernel.
                    cache.insert(graph_key.clone(), arc_kernel.clone());
                    (arc_kernel, details)
                }
            } else {
                log::debug!("Backend cache miss, compiling with deterministic opts...");
                let (new_kernel, details) = self.compile_kernel(graph_to_use, false);
                let arc_kernel = Arc::new(Mutex::new(new_kernel));
                *self.compile_count.lock().unwrap() += 1;
                cache.insert(graph_key.clone(), arc_kernel.clone());
                (arc_kernel, details)
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

        // Create the final map from NodeId to Buffer
        let mut output_map = HashMap::new();
        for (node_id, buffer_index) in &details.buffer_map {
            if *buffer_index < result_buffers.len() {
                output_map.insert(*node_id, result_buffers[*buffer_index].clone());
            }
        }
        output_map
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
        let result1 = backend.execute(&graph, vec![], vec![]);
        assert!(!result1.is_empty());
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            1,
            "First call should compile"
        );

        // Subsequent calls before the threshold should use the cache.
        let result2 = backend.execute(&graph, vec![], vec![]);
        assert!(!result2.is_empty());
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            1,
            "Second call should hit cache"
        );

        // The call at the threshold should trigger heuristic optimization and re-compilation.
        let result3 = backend.execute(&graph, vec![], vec![]);
        assert!(!result3.is_empty());
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            2,
            "Third call (at threshold) should recompile with heuristics"
        );

        // Calls after optimization should use the new heuristically optimized kernel from cache.
        let result4 = backend.execute(&graph, vec![], vec![]);
        assert!(!result4.is_empty());
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            2,
            "Fourth call should hit heuristic cache"
        );
        let result5 = backend.execute(&graph, vec![], vec![]);
        assert!(!result5.is_empty());
        assert_eq!(
            *backend.compile_count.lock().unwrap(),
            2,
            "Fifth call should hit heuristic cache"
        );
    }
}
