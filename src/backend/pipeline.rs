//! GPU Pipeline implementation
//!
//! This module provides a Pipeline implementation that uses the GPU backends
//! (OpenCL via `ocl` crate, Metal via `metal` crate).

use crate::ast::{AstNode, Literal};
use crate::backend::DeviceFeature;
use crate::backend::KernelSignature;
use crate::backend::renderer::CLikeRenderer;
use crate::backend::traits::{Compiler, Device, KernelConfig};
use crate::opt::ast::rules::rules_for_capabilities;
use crate::opt::ast::{
    AstSuggester, BeamSearchOptimizer, CompositeSuggester as AstCompositeSuggester,
    FunctionInliningSuggester, GroupParallelizationSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
    OptimizationHistory as AstOptimizationHistory, RuleBaseSuggester,
};
use crate::opt::context::DeviceCapabilities;
use crate::opt::progress::{IndicatifProgress, NoOpProgress};
use std::collections::HashMap;

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
    /// Beam width for AST optimization
    pub ast_beam_width: usize,
    /// Maximum optimization steps per phase
    pub max_steps: usize,
    /// Show progress during optimization
    pub show_progress: bool,
    /// Collect optimization history
    pub collect_history: bool,
    /// Enable fast math optimizations
    ///
    /// When enabled, the compiler may use faster but potentially less precise
    /// math operations (e.g., -cl-fast-relaxed-math for OpenCL).
    /// This can improve performance but may affect numerical precision.
    pub fast_math: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            ast_beam_width: 4,
            max_steps: 5000,
            show_progress: false,
            collect_history: cfg!(debug_assertions),
            fast_math: false, // デフォルトは無効（精度優先）
        }
    }
}

impl PipelineConfig {
    /// Enable fast math optimizations
    pub fn with_fast_math(mut self, enabled: bool) -> Self {
        self.fast_math = enabled;
        self
    }
}

/// Optimization histories for pipeline
#[derive(Debug, Clone, Default)]
pub struct OptimizationHistories {
    /// AST optimization history
    pub ast: Option<AstOptimizationHistory>,
}

/// GPU Pipeline for kernel compilation and execution
///
/// This pipeline uses GPU APIs (via `ocl` or `metal` crates) directly
/// from Rust, eliminating the need for external C host code generation.
///
/// # Type Parameters
/// * `R` - Renderer type (must implement KernelSourceRenderer)
/// * `Dev` - GPU device type
/// * `Comp` - GPU compiler type
pub struct Pipeline<R, Dev, Comp>
where
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
{
    renderer: R,
    compiler: Comp,
    device: Dev,
    config: PipelineConfig,
    /// Optimization histories
    pub histories: OptimizationHistories,
}

impl<R, Dev, Comp> Pipeline<R, Dev, Comp>
where
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
{
    /// Create a new pipeline
    pub fn new(renderer: R, compiler: Comp, device: Dev) -> Self {
        Self {
            renderer,
            compiler,
            device,
            config: PipelineConfig::default(),
            histories: OptimizationHistories::default(),
        }
    }

    /// Get reference to the device
    pub fn device(&self) -> &Dev {
        &self.device
    }

    /// Get mutable reference to the config
    pub fn config_mut(&mut self) -> &mut PipelineConfig {
        &mut self.config
    }

    /// Compile an AST program to a kernel
    ///
    /// This method compiles an AST generated from TensorLowerer.
    /// Returns a CacheEntry that can be stored in the kernel cache.
    pub fn compile_ast(
        &mut self,
        program: AstNode,
        signature: KernelSignature,
    ) -> Result<crate::backend::cache::CacheEntry, Comp::Error>
    where
        Comp::Kernel: 'static,
    {
        // Optimize AST
        let optimized_program = self.optimize_ast(program);

        // Render kernel source
        let kernel_source = self.renderer.render_kernel_source(&optimized_program);

        // Debug: log the kernel source
        #[cfg(debug_assertions)]
        eprintln!("[compile_ast] Generated kernel source:\n{}", kernel_source);

        log::debug!("Generated kernel source (from AST):\n{}", kernel_source);

        // Extract dispatch size config (for dynamic shape support)
        let dispatch_config = self.extract_dispatch_config(&optimized_program, &signature);

        // Extract the actual kernel/function name from the program
        let entry_point = self.extract_entry_point_name(&optimized_program);
        let base_config = self.extract_kernel_config(&optimized_program, &signature);

        // Compute compatible local_work_size
        let gws = base_config.global_work_size;
        let lws_base = base_config.local_work_size.unwrap_or([1, 1, 1]);

        // Get device max work group size
        let max_wg_size = self.device.profile().max_work_group_size;

        // Compute compatible local_work_size for each dimension
        let mut lws = [
            Self::compute_compatible_local_size(gws[0], lws_base[0]),
            Self::compute_compatible_local_size(gws[1], lws_base[1]),
            Self::compute_compatible_local_size(gws[2], lws_base[2]),
        ];

        // Ensure total work items don't exceed max_work_group_size
        while lws[0] * lws[1] * lws[2] > max_wg_size {
            // Find the largest dimension and reduce it
            let max_idx = if lws[0] >= lws[1] && lws[0] >= lws[2] {
                0
            } else if lws[1] >= lws[2] {
                1
            } else {
                2
            };
            // Find next smaller divisor of gws[max_idx]
            let mut new_val = lws[max_idx] - 1;
            while new_val > 1 && !gws[max_idx].is_multiple_of(new_val) {
                new_val -= 1;
            }
            lws[max_idx] = new_val.max(1);
        }

        log::debug!(
            "Kernel dispatch: global_work_size={:?}, local_work_size={:?} (base={:?}, max_wg_size={})",
            gws,
            lws,
            lws_base,
            max_wg_size
        );

        let kernel_config = KernelConfig::new(entry_point)
            .with_global_work_size(gws)
            .with_local_work_size(lws);

        // Compile kernel
        let kernel = self
            .compiler
            .compile(&self.device, &kernel_source, kernel_config)?;

        Ok(crate::backend::cache::CacheEntry::new(
            Box::new(kernel),
            signature,
            dispatch_config,
        ))
    }

    // Internal: optimize AST
    fn optimize_ast(&mut self, program: AstNode) -> AstNode {
        let opt_context = DeviceCapabilities::from_device(&self.device);

        // デバイス能力に基づいてルールを取得（代数簡約、定数畳み込み、FMA化等）
        let rules = rules_for_capabilities(&opt_context);

        // 利用可能なSIMD幅を取得
        let simd_widths = opt_context.all_simd_widths();

        // 全てのSuggesterを統合してビームサーチで探索
        let mut suggesters: Vec<Box<dyn AstSuggester>> = vec![
            // ルールベース（代数変換、定数畳み込み、FMA化）
            Box::new(RuleBaseSuggester::new(rules)),
            // ループ変換
            Box::new(LoopTilingSuggester::new()),
            Box::new(LoopInliningSuggester::new()),
            Box::new(LoopInterchangeSuggester::new()),
            Box::new(LoopFusionSuggester::new()),
            Box::new(FunctionInliningSuggester::with_default_limit()),
        ];

        // デバイスがSIMD対応していれば追加
        // TODO: Vectorization is disabled because it incorrectly vectorizes loads
        // from scalar buffers during broadcast/expand operations.
        // This needs to be fixed by adding shape-awareness to the vectorization pass.
        if !simd_widths.is_empty() {
            // suggesters.push(Box::new(VectorizationSuggester::with_widths(simd_widths)));
            let _ = simd_widths; // suppress unused warning
        }

        // デバイスが並列カーネルをサポートしていれば並列化Suggesterを追加
        if opt_context.supports_feature(DeviceFeature::ParallelKernel) {
            suggesters.push(Box::new(GroupParallelizationSuggester::new()));
            // TODO: LocalParallelizationは2次元以上で問題があるため、一時的に無効化
            // suggesters.push(Box::new(LocalParallelizationSuggester::new()));
        }

        let suggester = AstCompositeSuggester::new(suggesters);
        let (optimized, history) = if self.config.show_progress {
            let mut optimizer = BeamSearchOptimizer::new(suggester)
                .with_beam_width(self.config.ast_beam_width)
                .with_max_steps(self.config.max_steps)
                .with_progress(IndicatifProgress::new());
            optimizer.optimize_with_history(program)
        } else {
            let mut optimizer = BeamSearchOptimizer::new(suggester)
                .with_beam_width(self.config.ast_beam_width)
                .with_max_steps(self.config.max_steps)
                .with_progress(NoOpProgress::new());
            optimizer.optimize_with_history(program)
        };

        if self.config.collect_history {
            self.histories.ast = Some(history);
        }

        optimized
    }

    // Internal: extract entry point name from AST
    fn extract_entry_point_name(&self, program: &AstNode) -> String {
        if let AstNode::Program { functions, .. } = program {
            // First try to find a Kernel node with a name
            for f in functions {
                if let AstNode::Kernel { name: Some(n), .. } = f {
                    return n.clone();
                }
            }
            // Then try to find a Function node with a name
            for f in functions {
                if let AstNode::Function { name: Some(n), .. } = f {
                    return n.clone();
                }
            }
        }
        // Fallback to default name
        "main".to_string()
    }

    // Internal: extract dispatch size config from AST (for dynamic shape support)
    fn extract_dispatch_config(
        &self,
        program: &AstNode,
        signature: &KernelSignature,
    ) -> DispatchSizeConfig {
        // Find the first kernel function in the program
        if let AstNode::Program { functions, .. } = program {
            for f in functions {
                if let AstNode::Kernel {
                    default_grid_size,
                    default_thread_group_size,
                    ..
                } = f
                {
                    return DispatchSizeConfig::from_ast(
                        default_grid_size,
                        default_thread_group_size,
                    );
                }
            }
        }

        // Fallback: use output shape as grid size
        let total_elements: usize = signature
            .outputs
            .iter()
            .flat_map(|s| {
                s.shape.iter().filter_map(|e| {
                    if let crate::shape::Expr::Const(n) = e {
                        Some(*n as usize)
                    } else {
                        None
                    }
                })
            })
            .product::<usize>()
            .max(1);

        DispatchSizeConfig::from_const([total_elements, 1, 1], [1, 1, 1])
    }

    // Internal: extract kernel config from AST
    fn extract_kernel_config(
        &self,
        program: &AstNode,
        signature: &KernelSignature,
    ) -> KernelConfig {
        // Find the first kernel function in the program
        let first_kernel = if let AstNode::Program { functions, .. } = program {
            functions.iter().find_map(|f| {
                if let AstNode::Kernel { name: Some(n), .. } = f {
                    Some((n.clone(), f))
                } else {
                    None
                }
            })
        } else {
            None
        };

        // Try to extract grid/thread_group size from first Kernel node
        if let Some((
            kernel_name,
            AstNode::Kernel {
                default_grid_size,
                default_thread_group_size,
                ..
            },
        )) = first_kernel
        {
            let shape_vars = Self::extract_shape_vars(signature);
            let grid = evaluate_dispatch_size(default_grid_size, &shape_vars);
            let tg = evaluate_dispatch_size(default_thread_group_size, &shape_vars);

            // OpenCL: global_work_size = grid_size * thread_group_size
            // - grid_size: number of work groups (GroupId range)
            // - thread_group_size: threads per work group (LocalId range)
            let global_work_size = [grid[0] * tg[0], grid[1] * tg[1], grid[2] * tg[2]];

            let mut config = KernelConfig::new(kernel_name)
                .with_global_work_size(global_work_size)
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
                    if let crate::shape::Expr::Const(n) = e {
                        Some(*n as usize)
                    } else {
                        None
                    }
                })
            })
            .product::<usize>()
            .max(1);

        KernelConfig::new("main").with_global_work_size([total_elements, 1, 1])
    }

    /// Get shape variables from kernel signature
    ///
    /// Extracts symbolic shape variables from the signature's input and output shapes.
    /// The values are initialized to 0 and should be provided at runtime via
    /// ExecutionQuery::shape_var().
    fn extract_shape_vars(signature: &KernelSignature) -> HashMap<String, i64> {
        use std::collections::HashSet;

        fn collect_from_expr(expr: &crate::shape::Expr, vars: &mut HashSet<String>) {
            match expr {
                crate::shape::Expr::Sym(name) => {
                    vars.insert(name.clone());
                }
                crate::shape::Expr::Add(l, r)
                | crate::shape::Expr::Sub(l, r)
                | crate::shape::Expr::Mul(l, r)
                | crate::shape::Expr::Div(l, r)
                | crate::shape::Expr::Rem(l, r)
                | crate::shape::Expr::Lt(l, r)
                | crate::shape::Expr::And(l, r) => {
                    collect_from_expr(l, vars);
                    collect_from_expr(r, vars);
                }
                crate::shape::Expr::Not(a) => {
                    collect_from_expr(a, vars);
                }
                crate::shape::Expr::LoadIndex { offset_expr, .. } => {
                    collect_from_expr(offset_expr, vars);
                }
                crate::shape::Expr::Const(_)
                | crate::shape::Expr::Bool(_)
                | crate::shape::Expr::Idx(_) => {}
            }
        }

        let mut vars = HashSet::new();

        // Collect from input shapes
        for buf in &signature.inputs {
            for expr in &buf.shape {
                collect_from_expr(expr, &mut vars);
            }
        }

        // Collect from output shapes
        for buf in &signature.outputs {
            for expr in &buf.shape {
                collect_from_expr(expr, &mut vars);
            }
        }

        // Initialize all shape vars with default value 0
        // Actual values will be provided at runtime
        vars.into_iter().map(|name| (name, 0)).collect()
    }

    /// Compute a compatible local work size for OpenCL
    ///
    /// OpenCL requires:
    /// 1. local_work_size[i] <= global_work_size[i]
    /// 2. global_work_size[i] % local_work_size[i] == 0
    ///
    /// This function finds the largest value <= preferred that divides global evenly.
    fn compute_compatible_local_size(global: usize, preferred: usize) -> usize {
        if global == 0 {
            return 1;
        }

        // Start from min(preferred, global) and find a divisor
        let mut local = preferred.min(global);
        while local > 1 {
            if global.is_multiple_of(local) {
                return local;
            }
            local -= 1;
        }
        1
    }
}

/// Evaluate dispatch size from AST expressions
fn evaluate_dispatch_size(
    size: &[Box<AstNode>; 3],
    shape_vars: &HashMap<String, i64>,
) -> [usize; 3] {
    [
        evaluate_ast_expr(&size[0], shape_vars).max(1) as usize,
        evaluate_ast_expr(&size[1], shape_vars).max(1) as usize,
        evaluate_ast_expr(&size[2], shape_vars).max(1) as usize,
    ]
}

/// Evaluate an AST expression to a numeric value
fn evaluate_ast_expr(ast: &AstNode, shape_vars: &HashMap<String, i64>) -> i64 {
    match ast {
        AstNode::Const(lit) => match lit {
            Literal::I8(n) => *n as i64,
            Literal::I16(n) => *n as i64,
            Literal::I32(n) => *n as i64,
            Literal::I64(n) => *n,
            Literal::U8(n) => *n as i64,
            Literal::U16(n) => *n as i64,
            Literal::U32(n) => *n as i64,
            Literal::U64(n) => *n as i64,
            Literal::F16(f) => f.to_f64() as i64,
            Literal::BF16(f) => f.to_f64() as i64,
            Literal::F32(f) => *f as i64,
            Literal::F64(f) => *f as i64,
            Literal::Bool(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            Literal::Complex32(re, _) => *re as i64,
            Literal::Complex64(re, _) => *re as i64,
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

/// Expression for computing dispatch size dynamically
///
/// This represents a simple expression tree that can be evaluated
/// with different shape variable values at runtime.
#[derive(Debug, Clone)]
pub enum DispatchSizeExpr {
    /// Constant value
    Const(i64),
    /// Variable reference
    Var(String),
    /// Addition
    Add(Box<DispatchSizeExpr>, Box<DispatchSizeExpr>),
    /// Multiplication
    Mul(Box<DispatchSizeExpr>, Box<DispatchSizeExpr>),
    /// Integer division
    Div(Box<DispatchSizeExpr>, Box<DispatchSizeExpr>),
    /// Remainder
    Rem(Box<DispatchSizeExpr>, Box<DispatchSizeExpr>),
    /// Maximum of two values
    Max(Box<DispatchSizeExpr>, Box<DispatchSizeExpr>),
}

impl DispatchSizeExpr {
    /// Evaluate the expression with the given shape variables
    pub fn evaluate(&self, shape_vars: &HashMap<String, i64>) -> i64 {
        match self {
            Self::Const(n) => *n,
            Self::Var(name) => shape_vars.get(name).copied().unwrap_or(1),
            Self::Add(a, b) => a.evaluate(shape_vars) + b.evaluate(shape_vars),
            Self::Mul(a, b) => a.evaluate(shape_vars) * b.evaluate(shape_vars),
            Self::Div(a, b) => {
                let divisor = b.evaluate(shape_vars);
                if divisor != 0 {
                    a.evaluate(shape_vars) / divisor
                } else {
                    1
                }
            }
            Self::Rem(a, b) => {
                let divisor = b.evaluate(shape_vars);
                if divisor != 0 {
                    a.evaluate(shape_vars) % divisor
                } else {
                    0
                }
            }
            Self::Max(a, b) => a.evaluate(shape_vars).max(b.evaluate(shape_vars)),
        }
    }

    /// Create from AstNode
    pub fn from_ast(ast: &AstNode) -> Self {
        match ast {
            AstNode::Const(lit) => match lit {
                Literal::I8(n) => Self::Const(*n as i64),
                Literal::I16(n) => Self::Const(*n as i64),
                Literal::I32(n) => Self::Const(*n as i64),
                Literal::I64(n) => Self::Const(*n),
                Literal::U8(n) => Self::Const(*n as i64),
                Literal::U16(n) => Self::Const(*n as i64),
                Literal::U32(n) => Self::Const(*n as i64),
                Literal::U64(n) => Self::Const(*n as i64),
                Literal::F16(f) => Self::Const(f.to_f64() as i64),
                Literal::BF16(f) => Self::Const(f.to_f64() as i64),
                Literal::F32(f) => Self::Const(*f as i64),
                Literal::F64(f) => Self::Const(*f as i64),
                Literal::Bool(b) => Self::Const(if *b { 1 } else { 0 }),
                Literal::Complex32(re, _) => Self::Const(*re as i64),
                Literal::Complex64(re, _) => Self::Const(*re as i64),
            },
            AstNode::Var(name) => Self::Var(name.clone()),
            AstNode::Add(a, b) => {
                Self::Add(Box::new(Self::from_ast(a)), Box::new(Self::from_ast(b)))
            }
            AstNode::Mul(a, b) => {
                Self::Mul(Box::new(Self::from_ast(a)), Box::new(Self::from_ast(b)))
            }
            AstNode::Idiv(a, b) => {
                Self::Div(Box::new(Self::from_ast(a)), Box::new(Self::from_ast(b)))
            }
            AstNode::Rem(a, b) => {
                Self::Rem(Box::new(Self::from_ast(a)), Box::new(Self::from_ast(b)))
            }
            AstNode::Max(a, b) => {
                Self::Max(Box::new(Self::from_ast(a)), Box::new(Self::from_ast(b)))
            }
            _ => Self::Const(1), // Unsupported expressions default to 1
        }
    }
}

/// Grid and local size expressions for dynamic dispatch
#[derive(Debug, Clone)]
pub struct DispatchSizeConfig {
    /// Grid size expressions for each dimension
    pub grid_size: [DispatchSizeExpr; 3],
    /// Local size expressions for each dimension
    pub local_size: [DispatchSizeExpr; 3],
}

impl DispatchSizeConfig {
    /// Create with constant sizes
    pub fn from_const(grid: [usize; 3], local: [usize; 3]) -> Self {
        Self {
            grid_size: [
                DispatchSizeExpr::Const(grid[0] as i64),
                DispatchSizeExpr::Const(grid[1] as i64),
                DispatchSizeExpr::Const(grid[2] as i64),
            ],
            local_size: [
                DispatchSizeExpr::Const(local[0] as i64),
                DispatchSizeExpr::Const(local[1] as i64),
                DispatchSizeExpr::Const(local[2] as i64),
            ],
        }
    }

    /// Create from AST expressions
    pub fn from_ast(grid: &[Box<AstNode>; 3], local: &[Box<AstNode>; 3]) -> Self {
        Self {
            grid_size: [
                DispatchSizeExpr::from_ast(&grid[0]),
                DispatchSizeExpr::from_ast(&grid[1]),
                DispatchSizeExpr::from_ast(&grid[2]),
            ],
            local_size: [
                DispatchSizeExpr::from_ast(&local[0]),
                DispatchSizeExpr::from_ast(&local[1]),
                DispatchSizeExpr::from_ast(&local[2]),
            ],
        }
    }

    /// Evaluate grid size with shape variables
    pub fn evaluate_grid_size(&self, shape_vars: &HashMap<String, i64>) -> [usize; 3] {
        [
            self.grid_size[0].evaluate(shape_vars).max(1) as usize,
            self.grid_size[1].evaluate(shape_vars).max(1) as usize,
            self.grid_size[2].evaluate(shape_vars).max(1) as usize,
        ]
    }

    /// Evaluate local size with shape variables
    pub fn evaluate_local_size(&self, shape_vars: &HashMap<String, i64>) -> [usize; 3] {
        [
            self.local_size[0].evaluate(shape_vars).max(1) as usize,
            self.local_size[1].evaluate(shape_vars).max(1) as usize,
            self.local_size[2].evaluate(shape_vars).max(1) as usize,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::const_int;

    // Basic test to ensure the module compiles
    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
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
    fn test_dispatch_size_expr_const() {
        let expr = DispatchSizeExpr::Const(42);
        let shape_vars = HashMap::new();
        assert_eq!(expr.evaluate(&shape_vars), 42);
    }

    #[test]
    fn test_dispatch_size_config_from_const() {
        let config = DispatchSizeConfig::from_const([256, 1, 1], [64, 1, 1]);
        let shape_vars = HashMap::new();

        assert_eq!(config.evaluate_grid_size(&shape_vars), [256, 1, 1]);
        assert_eq!(config.evaluate_local_size(&shape_vars), [64, 1, 1]);
    }
}
