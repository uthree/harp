//! Unified compilation pipeline for CLI and Tensor::realize
//!
//! This module provides a common compilation pipeline that can be used by both
//! the CLI transpiler and the runtime execution path.
//!
//! # Example
//!
//! ```ignore
//! use eclat::backend::compile::{CompilationPipeline, OptimizationConfig};
//! use eclat::backend::DeviceKind;
//! use eclat_backend_cuda::CudaRenderer;
//!
//! // For CLI: render to code string (renderer provided by caller)
//! let pipeline = CompilationPipeline::new(DeviceKind::Cuda)
//!     .with_optimization(OptimizationConfig::level(2));
//! let ast = pipeline.lower(&[graph_node]);
//! let optimized = pipeline.optimize(ast);
//! let code = pipeline.render(&optimized, CudaRenderer::new());
//!
//! // For runtime: compile to kernel
//! let pipeline = CompilationPipeline::from_default_device();
//! let ast = pipeline.lower(&[graph_node]);
//! let kernel = pipeline.compile(ast, signature)?;
//! ```

use crate::ast::AstNode;
use crate::backend::renderer::Renderer;
use crate::backend::{DeviceKind, KernelSignature};
use crate::graph::GraphNode;
use crate::lowerer::Lowerer;
use crate::opt::ast::rules::all_algebraic_rules;
use crate::opt::ast::{
    AstOptimizer, AstSuggester, BeamSearchOptimizer, CompositeSuggester, FunctionInliningSuggester,
    GroupParallelizationSuggester, LocalParallelizationSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseSuggester,
    SharedMemorySuggester,
};
// Note: WmmaSuggester removed - WMMA detection is now done at graph level

/// Configuration for AST optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Optimization level (0-3)
    /// - 0: No optimization
    /// - 1: Basic optimizations (beam_width=4, max_steps=50)
    /// - 2: Moderate optimizations (beam_width=8, max_steps=100)
    /// - 3: Aggressive optimizations (beam_width=16, max_steps=200)
    pub level: u8,

    /// Beam search width (number of candidates to keep)
    pub beam_width: usize,

    /// Maximum optimization steps
    pub max_steps: usize,

    /// Whether to enable GPU parallelization suggesters
    pub enable_gpu_parallelization: bool,

    /// Whether to show progress during optimization
    pub show_progress: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::level(1)
    }
}

impl OptimizationConfig {
    /// Create optimization config from level (0-3)
    pub fn level(level: u8) -> Self {
        let (beam_width, max_steps) = match level {
            0 => (0, 0),
            1 => (4, 50),
            2 => (8, 100),
            3 => (16, 200),
            _ => (4, 50),
        };

        Self {
            level,
            beam_width,
            max_steps,
            enable_gpu_parallelization: false,
            show_progress: false,
        }
    }

    /// Enable GPU parallelization suggesters
    pub fn with_gpu_parallelization(mut self) -> Self {
        self.enable_gpu_parallelization = true;
        self
    }

    /// Enable progress display
    pub fn with_progress(mut self) -> Self {
        self.show_progress = true;
        self
    }

    /// Set custom beam width
    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
        self
    }

    /// Set custom max steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

/// Unified compilation pipeline
///
/// Provides methods for lowering, optimizing, rendering, and compiling
/// computation graphs to executable kernels.
#[derive(Debug, Clone)]
pub struct CompilationPipeline {
    /// Target backend
    backend: DeviceKind,

    /// Optimization configuration
    opt_config: OptimizationConfig,
}

impl CompilationPipeline {
    /// Create a new compilation pipeline for the specified backend
    pub fn new(backend: DeviceKind) -> Self {
        let mut opt_config = OptimizationConfig::default();

        // Enable GPU parallelization for GPU backends
        if matches!(
            backend,
            DeviceKind::Metal | DeviceKind::Cuda | DeviceKind::OpenCL
        ) {
            opt_config.enable_gpu_parallelization = true;
        }

        Self {
            backend,
            opt_config,
        }
    }

    /// Create a pipeline using the default device
    pub fn from_default_device() -> Self {
        let backend = crate::backend::get_default_device_kind();
        Self::new(backend)
    }

    /// Set the optimization configuration
    pub fn with_optimization(mut self, config: OptimizationConfig) -> Self {
        // Preserve GPU parallelization setting based on backend
        let enable_gpu = config.enable_gpu_parallelization
            || matches!(
                self.backend,
                DeviceKind::Metal | DeviceKind::Cuda | DeviceKind::OpenCL
            );
        self.opt_config = config;
        self.opt_config.enable_gpu_parallelization = enable_gpu;
        self
    }

    /// Set optimization level (0-3)
    pub fn with_opt_level(mut self, level: u8) -> Self {
        self.opt_config = OptimizationConfig::level(level);
        // Re-enable GPU parallelization for GPU backends
        if matches!(
            self.backend,
            DeviceKind::Metal | DeviceKind::Cuda | DeviceKind::OpenCL
        ) {
            self.opt_config.enable_gpu_parallelization = true;
        }
        self
    }

    /// Get the target backend
    pub fn backend(&self) -> DeviceKind {
        self.backend
    }

    /// Get the optimization configuration
    pub fn opt_config(&self) -> &OptimizationConfig {
        &self.opt_config
    }

    // ========================================================================
    // Pipeline stages
    // ========================================================================

    /// Lower GraphNode DAG to AST
    ///
    /// This is the first stage of compilation. It converts the computation
    /// graph into an AST representation suitable for optimization and code
    /// generation.
    ///
    /// When optimization level > 0, applies graph-level beam search optimization
    /// including MatMul pattern detection and fusion passes.
    ///
    /// # Errors
    ///
    /// Returns `LoweringError` if lowering fails.
    pub fn lower(&self, roots: &[GraphNode]) -> crate::lowerer::LoweringResult<AstNode> {
        let mut lowerer = Lowerer::new();
        if self.opt_config.level > 0 {
            lowerer.lower_with_graph_optimization(roots)
        } else {
            lowerer.lower(roots)
        }
    }

    /// Lower GraphNode DAG to AST, returning the Lowerer for buffer name lookups
    ///
    /// Use this when you need to map between GraphNode and buffer names.
    ///
    /// When optimization level > 0, applies graph-level beam search optimization
    /// including MatMul pattern detection and fusion passes.
    ///
    /// # Errors
    ///
    /// Returns `LoweringError` if lowering fails.
    pub fn lower_with_lowerer(
        &self,
        roots: &[GraphNode],
    ) -> crate::lowerer::LoweringResult<(AstNode, Lowerer)> {
        let mut lowerer = Lowerer::new();
        let ast = if self.opt_config.level > 0 {
            lowerer.lower_with_graph_optimization(roots)?
        } else {
            lowerer.lower(roots)?
        };
        Ok((ast, lowerer))
    }

    /// Optimize AST using beam search
    ///
    /// Applies various optimization passes including:
    /// - Algebraic simplifications (constant folding, identity removal)
    /// - Loop transformations (tiling, interchange, fusion)
    /// - Function inlining
    /// - GPU parallelization (for GPU backends)
    pub fn optimize(&self, ast: AstNode) -> AstNode {
        if self.opt_config.level == 0 {
            return ast;
        }

        let suggester = self.create_suggester();

        if self.opt_config.show_progress {
            let mut optimizer = BeamSearchOptimizer::new(suggester)
                .with_beam_width(self.opt_config.beam_width)
                .with_max_steps(self.opt_config.max_steps);
            optimizer.optimize(ast)
        } else {
            let mut optimizer = BeamSearchOptimizer::new(suggester)
                .with_beam_width(self.opt_config.beam_width)
                .with_max_steps(self.opt_config.max_steps)
                .without_progress();
            optimizer.optimize(ast)
        }
    }

    /// Optimize AST and return optimization history (for visualization)
    #[cfg(feature = "viz")]
    pub fn optimize_with_history(
        &self,
        ast: AstNode,
    ) -> (AstNode, crate::opt::ast::history::OptimizationHistory) {
        if self.opt_config.level == 0 {
            return (ast, crate::opt::ast::history::OptimizationHistory::new());
        }

        let suggester = self.create_suggester();
        let mut optimizer = BeamSearchOptimizer::new(suggester)
            .with_beam_width(self.opt_config.beam_width)
            .with_max_steps(self.opt_config.max_steps)
            .without_progress();
        optimizer.optimize_with_history(ast)
    }

    /// Render AST to source code string using the provided renderer
    ///
    /// The renderer should be appropriate for the target backend.
    /// Use backend-specific renderers from eclat-backend-* crates.
    pub fn render<R: Renderer>(&self, ast: &AstNode, renderer: R) -> String {
        renderer.render(ast).into()
    }

    /// Compile AST to executable kernel
    ///
    /// This compiles the AST using the default device and returns an
    /// executable kernel.
    pub fn compile(
        &self,
        ast: AstNode,
        signature: KernelSignature,
    ) -> Result<Box<dyn crate::backend::Kernel>, crate::backend::device::DeviceError> {
        // Optimize first
        let optimized = self.optimize(ast);
        crate::backend::compile_ast_on_default_device(optimized, signature)
    }

    /// Compile AST with caching support
    ///
    /// Returns a CacheEntry that includes the compiled kernel and dispatch
    /// configuration.
    pub fn compile_with_cache(
        &self,
        ast: AstNode,
        signature: KernelSignature,
    ) -> Result<crate::backend::CacheEntry, crate::backend::device::DeviceError> {
        // Optimize first
        let optimized = self.optimize(ast);
        crate::backend::global::compile_ast_with_cache(optimized, signature)
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    /// Create the composite suggester with configured suggesters
    fn create_suggester(&self) -> CompositeSuggester {
        let rules = all_algebraic_rules();
        let mut suggesters: Vec<Box<dyn AstSuggester>> = vec![
            Box::new(RuleBaseSuggester::new(rules)),
            Box::new(LoopTilingSuggester::new()),
            Box::new(LoopInliningSuggester::new()),
            Box::new(LoopInterchangeSuggester::new()),
            Box::new(LoopFusionSuggester::new()),
            Box::new(FunctionInliningSuggester::with_default_limit()),
        ];

        if self.opt_config.enable_gpu_parallelization {
            suggesters.push(Box::new(GroupParallelizationSuggester::new()));
            suggesters.push(Box::new(LocalParallelizationSuggester::new()));
            // Note: WMMA detection is now done at graph level (see MatMulDetectorSuggester)
            // 共有メモリ最適化 - GPU並列化時のみ有効
            suggesters.push(Box::new(SharedMemorySuggester::new()));
        }

        CompositeSuggester::new(suggesters)
    }

    /// Check if the backend is a GPU backend
    pub fn is_gpu_backend(&self) -> bool {
        matches!(
            self.backend,
            DeviceKind::Metal | DeviceKind::Cuda | DeviceKind::OpenCL
        )
    }
}

/// Mark outermost loops as parallel for OpenMP
///
/// This is used by the CLI when targeting the OpenMP backend.
pub fn mark_parallel_for_openmp(ast: AstNode) -> AstNode {
    mark_parallel_recursive(ast, true)
}

fn mark_parallel_recursive(ast: AstNode, is_outermost: bool) -> AstNode {
    use crate::ast::{ParallelInfo, ParallelKind};

    match ast {
        AstNode::Program {
            functions,
            execution_waves,
        } => {
            let functions = functions
                .into_iter()
                .map(|f| mark_parallel_recursive(f, true))
                .collect();
            AstNode::Program {
                functions,
                execution_waves,
            }
        }
        AstNode::Kernel {
            name,
            params,
            return_type,
            body,
            default_grid_size,
            default_thread_group_size,
        } => AstNode::Kernel {
            name,
            params,
            return_type,
            body: Box::new(mark_parallel_recursive(*body, true)),
            default_grid_size,
            default_thread_group_size,
        },
        AstNode::Range {
            var,
            start,
            step,
            stop,
            body,
            parallel,
        } => {
            let new_parallel = if is_outermost {
                ParallelInfo {
                    is_parallel: true,
                    kind: ParallelKind::OpenMP,
                    reductions: parallel.reductions,
                }
            } else {
                parallel
            };
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body: Box::new(mark_parallel_recursive(*body, false)),
                parallel: new_parallel,
            }
        }
        AstNode::Block { statements, scope } => {
            let statements = statements
                .into_iter()
                .map(|s| mark_parallel_recursive(s, is_outermost))
                .collect();
            AstNode::Block { statements, scope }
        }
        AstNode::If {
            condition,
            then_body,
            else_body,
        } => AstNode::If {
            condition,
            then_body: Box::new(mark_parallel_recursive(*then_body, false)),
            else_body: else_body.map(|e| Box::new(mark_parallel_recursive(*e, false))),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::{Expr, input};

    #[test]
    fn test_pipeline_lower() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let z = &x + &y;

        let pipeline = CompilationPipeline::new(DeviceKind::C);
        let ast = pipeline.lower(&[z]).expect("Lowering should succeed");

        assert!(matches!(ast, AstNode::Program { .. }));
    }

    #[test]
    fn test_pipeline_optimize() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = &x * &x;

        let pipeline = CompilationPipeline::new(DeviceKind::C).with_opt_level(1);
        let ast = pipeline.lower(&[y]).expect("Lowering should succeed");
        let optimized = pipeline.optimize(ast);

        assert!(matches!(optimized, AstNode::Program { .. }));
    }

    #[test]
    fn test_optimization_config_levels() {
        let config0 = OptimizationConfig::level(0);
        assert_eq!(config0.beam_width, 0);
        assert_eq!(config0.max_steps, 0);

        let config1 = OptimizationConfig::level(1);
        assert_eq!(config1.beam_width, 4);
        assert_eq!(config1.max_steps, 50);

        let config2 = OptimizationConfig::level(2);
        assert_eq!(config2.beam_width, 8);
        assert_eq!(config2.max_steps, 100);

        let config3 = OptimizationConfig::level(3);
        assert_eq!(config3.beam_width, 16);
        assert_eq!(config3.max_steps, 200);
    }

    #[test]
    fn test_gpu_parallelization_auto_enable() {
        let pipeline_c = CompilationPipeline::new(DeviceKind::C);
        assert!(!pipeline_c.opt_config().enable_gpu_parallelization);

        let pipeline_cuda = CompilationPipeline::new(DeviceKind::Cuda);
        assert!(pipeline_cuda.opt_config().enable_gpu_parallelization);

        let pipeline_metal = CompilationPipeline::new(DeviceKind::Metal);
        assert!(pipeline_metal.opt_config().enable_gpu_parallelization);
    }
}
