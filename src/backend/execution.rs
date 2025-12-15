//! GPU Pipeline implementation
//!
//! This module provides a Pipeline implementation that uses the GPU backends
//! (OpenCL via `ocl` crate, Metal via `metal` crate).

use crate::ast::program::KernelCall;
use crate::ast::{AstNode, DType, Literal};
use crate::backend::KernelSignature;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::sequence::{CompiledProgram, IntermediateBufferSpec, KernelCallInfo};
use crate::backend::traits::{
    KernelConfig, NativeBuffer, NativeCompiler, NativeContext, NativeKernel,
};
use crate::graph::Graph;
use crate::graph::shape::Expr;
use crate::opt::ast::rules::all_algebraic_rules;
use crate::opt::ast::{
    AstOptimizer, BeamSearchOptimizer as AstBeamSearchOptimizer,
    CompositeSuggester as AstCompositeSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
    OptimizationHistory as AstOptimizationHistory, RuleBaseOptimizer,
};
use crate::opt::graph::{GraphOptimizer, OptimizationHistory as GraphOptimizationHistory};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// Trait for renderers that can generate kernel-only source code
///
/// This trait extends CLikeRenderer to provide a method that generates
/// only the kernel source code (without host code), suitable for GPU APIs.
pub trait KernelSourceRenderer: CLikeRenderer {
    /// Render only the kernel source code (without host code)
    ///
    /// Returns the kernel function source that can be passed directly to
    /// GPU APIs (OpenCL, Metal).
    fn render_kernel_source(&mut self, program: &AstNode) -> String;
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Beam width for graph optimization
    pub graph_beam_width: usize,
    /// Beam width for AST optimization
    pub ast_beam_width: usize,
    /// Maximum optimization steps per phase
    pub max_steps: usize,
    /// Show progress during optimization
    pub show_progress: bool,
    /// Collect optimization history
    pub collect_history: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            graph_beam_width: 4,
            ast_beam_width: 4,
            max_steps: 5000,
            show_progress: false,
            collect_history: cfg!(debug_assertions),
        }
    }
}

/// Optimization histories for pipeline
#[derive(Debug, Clone, Default)]
pub struct OptimizationHistories {
    /// Graph optimization history
    pub graph: Option<GraphOptimizationHistory>,
    /// AST optimization history
    pub ast: Option<AstOptimizationHistory>,
}

/// GPU Pipeline for kernel compilation and execution
///
/// This pipeline uses GPU APIs (via `ocl` or `metal` crates) directly
/// from Rust, eliminating the need for C host code generation and libloading.
///
/// # Type Parameters
/// * `R` - Renderer type (must implement KernelSourceRenderer)
/// * `Ctx` - GPU context type
/// * `Comp` - GPU compiler type
pub struct Pipeline<R, Ctx, Comp>
where
    R: KernelSourceRenderer + Clone,
    Ctx: NativeContext,
    Comp: NativeCompiler<Context = Ctx>,
{
    renderer: R,
    compiler: Comp,
    context: Ctx,
    config: PipelineConfig,
    /// Optimization histories
    pub histories: OptimizationHistories,
    /// Compiled kernel cache
    kernel_cache: HashMap<String, Comp::Kernel>,
}

impl<R, Ctx, Comp, Buf> Pipeline<R, Ctx, Comp>
where
    R: KernelSourceRenderer + Clone,
    Ctx: NativeContext,
    Buf: NativeBuffer<Context = Ctx>,
    Comp: NativeCompiler<Context = Ctx>,
    Comp::Kernel: NativeKernel<Buffer = Buf> + Clone,
{
    /// Create a new pipeline
    pub fn new(renderer: R, compiler: Comp, context: Ctx) -> Self {
        Self {
            renderer,
            compiler,
            context,
            config: PipelineConfig::default(),
            histories: OptimizationHistories::default(),
            kernel_cache: HashMap::new(),
        }
    }

    /// Get reference to the context
    pub fn context(&self) -> &Ctx {
        &self.context
    }

    /// Get mutable reference to the config
    pub fn config_mut(&mut self) -> &mut PipelineConfig {
        &mut self.config
    }

    /// Compile a graph to a kernel
    pub fn compile_graph(
        &mut self,
        graph: Graph,
    ) -> Result<CompiledKernel<Comp::Kernel, Buf>, Comp::Error> {
        // Create signature from original graph
        let signature = crate::lowerer::create_signature(&graph);

        // Optimize graph
        let optimized_graph = self.optimize_graph(graph);

        // Extract program from graph
        let program = crate::lowerer::extract_program(optimized_graph);

        // Optimize AST
        let optimized_program = self.optimize_ast(program);

        // Render kernel source
        let kernel_source = self.renderer.render_kernel_source(&optimized_program);

        // Extract kernel config from the program
        let kernel_config = self.extract_kernel_config(&optimized_program, &signature);

        // Compile kernel
        let kernel = self
            .compiler
            .compile(&self.context, &kernel_source, kernel_config)?;

        Ok(CompiledKernel {
            kernel,
            signature,
            _buffer: PhantomData,
        })
    }

    /// Compile and cache a kernel
    pub fn compile_and_cache(
        &mut self,
        key: String,
        graph: Graph,
    ) -> Result<&Comp::Kernel, Comp::Error> {
        let compiled = self.compile_graph(graph)?;
        self.kernel_cache.insert(key.clone(), compiled.kernel);
        Ok(self.kernel_cache.get(&key).unwrap())
    }

    /// Get a cached kernel
    pub fn get_cached_kernel(&self, key: &str) -> Option<&Comp::Kernel> {
        self.kernel_cache.get(key)
    }

    /// Clear the kernel cache
    pub fn clear_cache(&mut self) {
        self.kernel_cache.clear();
    }

    /// Allocate a buffer on the GPU
    pub fn allocate_buffer(&self, shape: Vec<usize>, dtype: DType) -> Result<Buf, Buf::Error> {
        Buf::allocate(&self.context, shape, dtype)
    }

    /// Compile a graph to a program (supports multiple kernels)
    ///
    /// This method compiles a graph that may produce multiple kernels after optimization.
    /// The returned `CompiledProgram` can execute all kernels in the correct order.
    pub fn compile_program(
        &mut self,
        graph: Graph,
    ) -> Result<CompiledProgram<Comp::Kernel, Buf>, Comp::Error> {
        // Create signature from original graph
        let signature = crate::lowerer::create_signature(&graph);

        // Optimize graph
        let optimized_graph = self.optimize_graph(graph);

        // Extract program from graph
        let program = crate::lowerer::extract_program(optimized_graph);

        // Optimize AST
        let optimized_program = self.optimize_ast(program);

        // Render kernel source (all kernels together)
        let kernel_source = self.renderer.render_kernel_source(&optimized_program);

        // Extract program structure
        let AstNode::Program {
            functions,
            entry_point,
            execution_order,
        } = &optimized_program
        else {
            // Fallback for non-Program nodes: use compile_graph
            let compiled =
                self.compile_graph_internal(optimized_program, signature.clone(), kernel_source)?;
            // Use default entry point name
            let entry_name = "main".to_string();
            return Ok(CompiledProgram::new(
                HashMap::from([(entry_name, compiled.kernel)]),
                vec![],
                vec![],
                signature.inputs.iter().map(|i| i.name.clone()).collect(),
                signature.outputs.iter().map(|o| o.name.clone()).collect(),
            ));
        };

        // Collect all kernel names from the program
        let kernel_names: Vec<String> = functions
            .iter()
            .filter_map(|f| {
                if let AstNode::Kernel { name, .. } = f {
                    name.clone()
                } else {
                    None
                }
            })
            .collect();

        // Compile each kernel with its specific config
        let shape_vars = Self::extract_shape_vars(&signature);
        let mut kernels: HashMap<String, Comp::Kernel> = HashMap::new();

        for func in functions {
            if let AstNode::Kernel {
                name: Some(name),
                default_grid_size,
                default_thread_group_size,
                ..
            } = func
            {
                let grid = evaluate_dispatch_size(default_grid_size, &shape_vars);
                let tg = evaluate_dispatch_size(default_thread_group_size, &shape_vars);

                let mut config = KernelConfig::new(name.clone())
                    .with_global_work_size(grid)
                    .with_local_work_size(tg);

                for (var_name, value) in &shape_vars {
                    config = config.with_shape_var(var_name.clone(), *value);
                }

                let kernel = self
                    .compiler
                    .compile(&self.context, &kernel_source, config)?;
                kernels.insert(name.clone(), kernel);
            }
        }

        // If no kernels were found but entry_point exists, try to compile entry_point
        if kernels.is_empty() {
            let config = self.extract_kernel_config(&optimized_program, &signature);
            let kernel = self
                .compiler
                .compile(&self.context, &kernel_source, config)?;
            kernels.insert(entry_point.clone(), kernel);
        }

        // Convert execution_order to KernelCallInfo
        let call_infos = self.convert_execution_order(execution_order, &shape_vars);

        // If no execution order specified, create a single call to entry_point
        let call_infos = if call_infos.is_empty() && !kernels.is_empty() {
            let entry = kernels.keys().next().unwrap().clone();
            let config = kernels
                .get(&entry)
                .map(|k| k.config())
                .cloned()
                .unwrap_or_else(|| KernelConfig::new(&entry));

            vec![KernelCallInfo::new(
                entry,
                signature.inputs.iter().map(|i| i.name.clone()).collect(),
                signature.outputs.iter().map(|o| o.name.clone()).collect(),
                config.global_work_size,
                config.local_work_size.unwrap_or([1, 1, 1]),
            )]
        } else {
            call_infos
        };

        // Analyze intermediate buffers
        let intermediate_specs =
            self.analyze_intermediate_buffers(&call_infos, &signature, &kernel_names);

        Ok(CompiledProgram::new(
            kernels,
            call_infos,
            intermediate_specs,
            signature.inputs.iter().map(|i| i.name.clone()).collect(),
            signature.outputs.iter().map(|o| o.name.clone()).collect(),
        ))
    }

    // Internal: compile graph with provided AST and source
    fn compile_graph_internal(
        &mut self,
        program: AstNode,
        signature: KernelSignature,
        kernel_source: String,
    ) -> Result<CompiledKernel<Comp::Kernel, Buf>, Comp::Error> {
        let kernel_config = self.extract_kernel_config(&program, &signature);
        let kernel = self
            .compiler
            .compile(&self.context, &kernel_source, kernel_config)?;

        Ok(CompiledKernel {
            kernel,
            signature,
            _buffer: PhantomData,
        })
    }

    // Internal: convert execution_order to KernelCallInfo
    fn convert_execution_order(
        &self,
        execution_order: &[KernelCall],
        shape_vars: &HashMap<String, isize>,
    ) -> Vec<KernelCallInfo> {
        execution_order
            .iter()
            .map(|call| {
                let grid_size = evaluate_expr_vec(&call.grid_size, shape_vars);
                let local_size = evaluate_expr_vec(&call.thread_group_size, shape_vars);

                KernelCallInfo::new(
                    call.kernel_name.clone(),
                    call.inputs.clone(),
                    call.outputs.clone(),
                    grid_size,
                    local_size,
                )
            })
            .collect()
    }

    // Internal: analyze and identify intermediate buffers
    fn analyze_intermediate_buffers(
        &self,
        call_infos: &[KernelCallInfo],
        signature: &KernelSignature,
        _kernel_names: &[String],
    ) -> Vec<IntermediateBufferSpec> {
        // Collect all external input/output names
        let external_inputs: HashSet<_> = signature.inputs.iter().map(|i| i.name.clone()).collect();
        let external_outputs: HashSet<_> =
            signature.outputs.iter().map(|o| o.name.clone()).collect();

        // Collect all buffers used in calls
        let mut all_inputs: HashSet<String> = HashSet::new();
        let mut all_outputs: HashSet<String> = HashSet::new();

        for call in call_infos {
            for input in &call.inputs {
                all_inputs.insert(input.clone());
            }
            for output in &call.outputs {
                all_outputs.insert(output.clone());
            }
        }

        // Intermediate buffers are those that are produced (output) by one kernel
        // and consumed (input) by another, but are not external inputs/outputs
        let intermediate_names: HashSet<_> = all_outputs
            .intersection(&all_inputs)
            .filter(|name| !external_inputs.contains(*name) && !external_outputs.contains(*name))
            .cloned()
            .collect();

        // Create specs for intermediate buffers
        // For now, we try to infer shape from the signature or use a default
        // Note: BufferSignature doesn't have dtype, so we default to F32
        intermediate_names
            .into_iter()
            .map(|name| {
                // Try to find buffer shape from signature
                let shape = signature
                    .outputs
                    .iter()
                    .find(|o| o.name == name)
                    .map(|o| {
                        o.shape
                            .iter()
                            .map(|e| e.as_usize().unwrap_or(1))
                            .collect::<Vec<usize>>()
                    })
                    .or_else(|| {
                        signature.inputs.iter().find(|i| i.name == name).map(|i| {
                            i.shape
                                .iter()
                                .map(|e| e.as_usize().unwrap_or(1))
                                .collect::<Vec<usize>>()
                        })
                    })
                    .unwrap_or_else(|| vec![1]);

                // Default to F32 for intermediate buffers
                IntermediateBufferSpec::new(name, shape, DType::F32)
            })
            .collect()
    }

    // Internal: optimize graph
    fn optimize_graph(&mut self, graph: Graph) -> Graph {
        use crate::backend::pipeline::{MultiPhaseConfig, create_multi_phase_optimizer};

        let config = MultiPhaseConfig::new()
            .with_beam_width(self.config.graph_beam_width)
            .with_max_steps(self.config.max_steps)
            .with_progress(self.config.show_progress)
            .with_collect_logs(self.config.collect_history);

        let optimizer = create_multi_phase_optimizer(config);
        let (optimized_graph, history) = optimizer.optimize_with_history(graph);

        if self.config.collect_history {
            self.histories.graph = Some(history);
        }

        optimized_graph
    }

    // Internal: optimize AST
    fn optimize_ast(&mut self, program: AstNode) -> AstNode {
        // Phase 1: Rule-based optimization
        let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules());
        let rule_optimized = rule_optimizer.optimize(program);

        // Phase 2: Loop optimization with beam search
        let loop_suggester = AstCompositeSuggester::new(vec![
            Box::new(LoopTilingSuggester::new()),
            Box::new(LoopInliningSuggester::new()),
            Box::new(LoopInterchangeSuggester::new()),
            Box::new(LoopFusionSuggester::new()),
            Box::new(FunctionInliningSuggester::with_default_limit()),
        ]);

        let loop_optimizer = AstBeamSearchOptimizer::new(loop_suggester)
            .with_beam_width(self.config.ast_beam_width)
            .with_max_steps(self.config.max_steps)
            .with_progress(self.config.show_progress);

        let (optimized, history) = loop_optimizer.optimize_with_history(rule_optimized);

        if self.config.collect_history {
            self.histories.ast = Some(history);
        }

        optimized
    }

    // Internal: extract kernel config from AST
    fn extract_kernel_config(
        &self,
        program: &AstNode,
        signature: &KernelSignature,
    ) -> KernelConfig {
        // Extract entry point name
        let entry_point = if let AstNode::Program { entry_point, .. } = program {
            entry_point.clone()
        } else {
            "main".to_string()
        };

        // Try to extract grid/thread_group size from Kernel node
        if let Some(AstNode::Kernel {
            default_grid_size,
            default_thread_group_size,
            ..
        }) = program.get_function(&entry_point)
        {
            let shape_vars = Self::extract_shape_vars(signature);
            let grid = evaluate_dispatch_size(default_grid_size, &shape_vars);
            let tg = evaluate_dispatch_size(default_thread_group_size, &shape_vars);

            let mut config = KernelConfig::new(entry_point)
                .with_global_work_size(grid)
                .with_local_work_size(tg);

            // Add shape vars to config
            for (name, value) in shape_vars {
                config = config.with_shape_var(name, value);
            }

            return config;
        }

        // Fallback: Calculate global work size from output shapes
        let total_elements: usize = signature
            .outputs
            .iter()
            .flat_map(|s| {
                s.shape.iter().filter_map(|e| {
                    if let crate::graph::shape::Expr::Const(n) = e {
                        Some(*n as usize)
                    } else {
                        None
                    }
                })
            })
            .product::<usize>()
            .max(1);

        KernelConfig::new(entry_point).with_global_work_size([total_elements, 1, 1])
    }

    /// Extract shape variables from signature
    fn extract_shape_vars(signature: &KernelSignature) -> HashMap<String, isize> {
        signature.shape_vars.clone()
    }
}

/// Evaluate dispatch size from AST expressions
fn evaluate_dispatch_size(
    size: &[Box<AstNode>; 3],
    shape_vars: &HashMap<String, isize>,
) -> [usize; 3] {
    [
        evaluate_ast_expr(&size[0], shape_vars).max(1) as usize,
        evaluate_ast_expr(&size[1], shape_vars).max(1) as usize,
        evaluate_ast_expr(&size[2], shape_vars).max(1) as usize,
    ]
}

/// Evaluate an AST expression to a numeric value
fn evaluate_ast_expr(ast: &AstNode, shape_vars: &HashMap<String, isize>) -> isize {
    match ast {
        AstNode::Const(lit) => match lit {
            Literal::Int(n) => *n,
            Literal::F32(f) => *f as isize,
            Literal::Bool(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
        },
        AstNode::Var(name) => shape_vars.get(name).copied().unwrap_or(1),
        AstNode::Mul(a, b) => evaluate_ast_expr(a, shape_vars) * evaluate_ast_expr(b, shape_vars),
        AstNode::Add(a, b) => evaluate_ast_expr(a, shape_vars) + evaluate_ast_expr(b, shape_vars),
        AstNode::Idiv(a, b) => {
            let divisor = evaluate_ast_expr(b, shape_vars);
            if divisor != 0 {
                evaluate_ast_expr(a, shape_vars) / divisor
            } else {
                1
            }
        }
        AstNode::Rem(a, b) => {
            let divisor = evaluate_ast_expr(b, shape_vars);
            if divisor != 0 {
                evaluate_ast_expr(a, shape_vars) % divisor
            } else {
                0
            }
        }
        AstNode::Max(a, b) => {
            let va = evaluate_ast_expr(a, shape_vars);
            let vb = evaluate_ast_expr(b, shape_vars);
            va.max(vb)
        }
        _ => 1, // Unsupported expressions default to 1
    }
}

/// Evaluate a Vec<Expr> to [usize; 3] for dispatch sizes
fn evaluate_expr_vec(exprs: &[Expr], shape_vars: &HashMap<String, isize>) -> [usize; 3] {
    let mut result = [1usize; 3];
    for (i, expr) in exprs.iter().enumerate().take(3) {
        result[i] = evaluate_expr(expr, shape_vars).max(1) as usize;
    }
    result
}

/// Evaluate a shape Expr to a numeric value
fn evaluate_expr(expr: &Expr, shape_vars: &HashMap<String, isize>) -> isize {
    match expr {
        Expr::Const(n) => *n,
        Expr::Var(name) => shape_vars.get(name).copied().unwrap_or(1),
        Expr::Add(a, b) => evaluate_expr(a, shape_vars) + evaluate_expr(b, shape_vars),
        Expr::Sub(a, b) => evaluate_expr(a, shape_vars) - evaluate_expr(b, shape_vars),
        Expr::Mul(a, b) => evaluate_expr(a, shape_vars) * evaluate_expr(b, shape_vars),
        Expr::Div(a, b) => {
            let divisor = evaluate_expr(b, shape_vars);
            if divisor != 0 {
                evaluate_expr(a, shape_vars) / divisor
            } else {
                1
            }
        }
        Expr::Rem(a, b) => {
            let divisor = evaluate_expr(b, shape_vars);
            if divisor != 0 {
                evaluate_expr(a, shape_vars) % divisor
            } else {
                0
            }
        }
    }
}

/// A compiled kernel with its signature
pub struct CompiledKernel<K, B>
where
    K: NativeKernel<Buffer = B>,
    B: NativeBuffer,
{
    /// The compiled kernel
    pub kernel: K,
    /// The kernel signature
    pub signature: KernelSignature,
    _buffer: PhantomData<B>,
}

impl<K, B> CompiledKernel<K, B>
where
    K: NativeKernel<Buffer = B>,
    B: NativeBuffer,
{
    /// Execute the kernel with the given buffers
    pub fn execute(&self, inputs: &[&B], outputs: &mut [&mut B]) -> Result<(), K::Error> {
        self.kernel.execute(inputs, outputs)
    }

    /// Get the kernel signature
    pub fn signature(&self) -> &KernelSignature {
        &self.signature
    }
}

// Type aliases for backward compatibility
pub type NativePipeline<R, Ctx, Comp> = Pipeline<R, Ctx, Comp>;
pub type NativePipelineConfig = PipelineConfig;
pub type NativeOptimizationHistories = OptimizationHistories;
pub type CompiledNativeKernel<K, B> = CompiledKernel<K, B>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::const_int;

    // Basic test to ensure the module compiles
    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.graph_beam_width, 4);
        assert_eq!(config.ast_beam_width, 4);
    }

    #[test]
    fn test_evaluate_ast_expr_const() {
        let shape_vars = HashMap::new();
        assert_eq!(evaluate_ast_expr(&const_int(42), &shape_vars), 42);
        assert_eq!(evaluate_ast_expr(&const_int(0), &shape_vars), 0);
    }

    #[test]
    fn test_evaluate_ast_expr_var() {
        let mut shape_vars = HashMap::new();
        shape_vars.insert("n".to_string(), 128);

        let var_n = AstNode::Var("n".to_string());
        assert_eq!(evaluate_ast_expr(&var_n, &shape_vars), 128);

        // Unknown variable defaults to 1
        let var_unknown = AstNode::Var("unknown".to_string());
        assert_eq!(evaluate_ast_expr(&var_unknown, &shape_vars), 1);
    }

    #[test]
    fn test_evaluate_ast_expr_arithmetic() {
        let shape_vars = HashMap::new();

        // 10 * 20 = 200
        let mul = AstNode::Mul(Box::new(const_int(10)), Box::new(const_int(20)));
        assert_eq!(evaluate_ast_expr(&mul, &shape_vars), 200);

        // 10 + 20 = 30
        let add = AstNode::Add(Box::new(const_int(10)), Box::new(const_int(20)));
        assert_eq!(evaluate_ast_expr(&add, &shape_vars), 30);

        // 100 / 10 = 10
        let div = AstNode::Idiv(Box::new(const_int(100)), Box::new(const_int(10)));
        assert_eq!(evaluate_ast_expr(&div, &shape_vars), 10);
    }

    #[test]
    fn test_evaluate_dispatch_size() {
        let shape_vars = HashMap::new();
        let size: [Box<AstNode>; 3] = [
            Box::new(const_int(256)),
            Box::new(const_int(1)),
            Box::new(const_int(1)),
        ];

        let result = evaluate_dispatch_size(&size, &shape_vars);
        assert_eq!(result, [256, 1, 1]);
    }

    #[test]
    fn test_evaluate_dispatch_size_with_vars() {
        let mut shape_vars = HashMap::new();
        shape_vars.insert("total".to_string(), 1024);

        // grid_size = ceil(total / 64) * 64
        // Represented as: Mul(Idiv(Add(total, 63), 64), 64)
        let grid_x = AstNode::Mul(
            Box::new(AstNode::Idiv(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Var("total".to_string())),
                    Box::new(const_int(63)),
                )),
                Box::new(const_int(64)),
            )),
            Box::new(const_int(64)),
        );

        let size: [Box<AstNode>; 3] = [
            Box::new(grid_x),
            Box::new(const_int(1)),
            Box::new(const_int(1)),
        ];

        let result = evaluate_dispatch_size(&size, &shape_vars);
        // (1024 + 63) / 64 = 16, 16 * 64 = 1024
        assert_eq!(result, [1024, 1, 1]);
    }
}
