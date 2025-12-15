//! Native Pipeline implementation
//!
//! This module provides a Pipeline implementation that uses the native GPU backends
//! (OpenCL via `ocl` crate, Metal via `metal` crate) without libloading.

use crate::ast::{AstNode, DType};
use crate::backend::KernelSignature;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::native::{
    KernelConfig, NativeBuffer, NativeCompiler, NativeContext, NativeKernel,
};
use crate::graph::Graph;
use crate::opt::ast::rules::all_algebraic_rules;
use crate::opt::ast::{
    AstOptimizer, BeamSearchOptimizer as AstBeamSearchOptimizer,
    CompositeSuggester as AstCompositeSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
    OptimizationHistory as AstOptimizationHistory, RuleBaseOptimizer,
};
use crate::opt::graph::{GraphOptimizer, OptimizationHistory as GraphOptimizationHistory};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Trait for renderers that can generate kernel-only source code
///
/// This trait extends CLikeRenderer to provide a method that generates
/// only the kernel source code (without host code), suitable for native GPU APIs.
pub trait KernelSourceRenderer: CLikeRenderer {
    /// Render only the kernel source code (without host code)
    ///
    /// Returns the kernel function source that can be passed directly to
    /// native GPU APIs (OpenCL, Metal).
    fn render_kernel_source(&mut self, program: &AstNode) -> String;
}

/// Native Pipeline configuration
#[derive(Debug, Clone)]
pub struct NativePipelineConfig {
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

impl Default for NativePipelineConfig {
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

/// Optimization histories for native pipeline
#[derive(Debug, Clone, Default)]
pub struct NativeOptimizationHistories {
    /// Graph optimization history
    pub graph: Option<GraphOptimizationHistory>,
    /// AST optimization history
    pub ast: Option<AstOptimizationHistory>,
}

/// Native Pipeline for GPU execution without libloading
///
/// This pipeline uses native GPU APIs (via `ocl` or `metal` crates) directly
/// from Rust, eliminating the need for C host code generation and libloading.
///
/// # Type Parameters
/// * `R` - Renderer type (must implement KernelSourceRenderer)
/// * `Ctx` - GPU context type
/// * `Comp` - GPU compiler type
pub struct NativePipeline<R, Ctx, Comp>
where
    R: KernelSourceRenderer + Clone,
    Ctx: NativeContext,
    Comp: NativeCompiler<Context = Ctx>,
{
    renderer: R,
    compiler: Comp,
    context: Ctx,
    config: NativePipelineConfig,
    /// Optimization histories
    pub histories: NativeOptimizationHistories,
    /// Compiled kernel cache
    kernel_cache: HashMap<String, Comp::Kernel>,
}

impl<R, Ctx, Comp, Buf> NativePipeline<R, Ctx, Comp>
where
    R: KernelSourceRenderer + Clone,
    Ctx: NativeContext,
    Buf: NativeBuffer<Context = Ctx>,
    Comp: NativeCompiler<Context = Ctx>,
    Comp::Kernel: NativeKernel<Buffer = Buf> + Clone,
{
    /// Create a new native pipeline
    pub fn new(renderer: R, compiler: Comp, context: Ctx) -> Self {
        Self {
            renderer,
            compiler,
            context,
            config: NativePipelineConfig::default(),
            histories: NativeOptimizationHistories::default(),
            kernel_cache: HashMap::new(),
        }
    }

    /// Get reference to the context
    pub fn context(&self) -> &Ctx {
        &self.context
    }

    /// Get mutable reference to the config
    pub fn config_mut(&mut self) -> &mut NativePipelineConfig {
        &mut self.config
    }

    /// Compile a graph to a native kernel
    pub fn compile_graph(
        &mut self,
        graph: Graph,
    ) -> Result<CompiledNativeKernel<Comp::Kernel, Buf>, Comp::Error> {
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

        Ok(CompiledNativeKernel {
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

        // Calculate global work size from output shapes
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

        // Use 1D global work size for simplicity
        // More sophisticated work size calculation could be done based on AST analysis
        KernelConfig::new(entry_point).with_global_work_size([total_elements, 1, 1])
    }
}

/// A compiled native kernel with its signature
pub struct CompiledNativeKernel<K, B>
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

impl<K, B> CompiledNativeKernel<K, B>
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

#[cfg(test)]
mod tests {
    use super::*;

    // Basic test to ensure the module compiles
    #[test]
    fn test_native_pipeline_config_default() {
        let config = NativePipelineConfig::default();
        assert_eq!(config.graph_beam_width, 4);
        assert_eq!(config.ast_beam_width, 4);
    }
}
