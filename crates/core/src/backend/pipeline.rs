//! GPU Pipeline implementation
//!
//! This module provides a Pipeline implementation that uses the GPU backends
//! (OpenCL via `ocl` crate, Metal via `metal` crate).

use crate::ast::{AstKernelCallInfo, AstNode, DType, Literal};
use crate::backend::KernelSignature;
use crate::backend::c_like::CLikeRenderer;
use crate::backend::sequence::{
    CompiledProgram, ExecutionQuery, IntermediateBufferSpec, KernelCallInfo,
};
use crate::backend::traits::{Buffer, Compiler, Device, Kernel, KernelConfig};
use crate::graph::Graph;
use crate::opt::ast::rules::rules_for_capabilities;
use crate::opt::ast::{
    AstSuggester, BeamSearchOptimizer as AstBeamSearchOptimizer,
    CompositeSuggester as AstCompositeSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester,
    OptimizationHistory as AstOptimizationHistory, RuleBaseSuggester, VectorizationSuggester,
};
use crate::opt::context::DeviceCapabilities;
use crate::opt::graph::{GraphOptimizer, OptimizationHistory as GraphOptimizationHistory};
use std::collections::HashMap;
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
            graph_beam_width: 4,
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
    /// Graph optimization history
    pub graph: Option<GraphOptimizationHistory>,
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

impl<R, Dev, Comp, Buf> Pipeline<R, Dev, Comp>
where
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Buf: Buffer<Dev = Dev>,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
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

        // Debug: log the kernel source
        log::debug!("Generated kernel source:\n{}", kernel_source);

        // Extract dispatch size config (for dynamic shape support)
        let dispatch_config = self.extract_dispatch_config(&optimized_program, &signature);

        // Extract the actual kernel/function name from the program
        let entry_point = self.extract_entry_point_name(&optimized_program);
        let base_config = self.extract_kernel_config(&optimized_program, &signature);

        // Compute compatible local_work_size:
        // local_work_size[i] must be <= global_work_size[i] and must divide evenly
        let gws = base_config.global_work_size;
        let lws_base = base_config.local_work_size.unwrap_or([1, 1, 1]);
        let lws = [
            Self::compute_compatible_local_size(gws[0], lws_base[0]),
            Self::compute_compatible_local_size(gws[1], lws_base[1]),
            Self::compute_compatible_local_size(gws[2], lws_base[2]),
        ];

        let kernel_config = KernelConfig::new(entry_point)
            .with_global_work_size(gws)
            .with_local_work_size(lws);

        // Compile kernel
        let kernel = self
            .compiler
            .compile(&self.device, &kernel_source, kernel_config)?;

        Ok(CompiledKernel {
            kernel,
            signature,
            dispatch_config,
            _buffer: PhantomData,
        })
    }

    /// Allocate a buffer on the GPU
    pub fn allocate_buffer(&self, shape: Vec<usize>, dtype: DType) -> Result<Buf, Buf::Error> {
        Buf::allocate(&self.device, shape, dtype)
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
        let AstNode::Program { functions, .. } = &optimized_program else {
            // Fallback for non-Program nodes: use compile_graph
            let compiled =
                self.compile_graph_internal(optimized_program, signature.clone(), kernel_source)?;
            // Use default entry point name
            let entry_name = "main".to_string();
            return Ok(CompiledProgram::new(
                HashMap::from([(entry_name, compiled.kernel)]),
                vec![], // empty execution_waves
                vec![],
                signature.inputs.iter().map(|i| i.name.clone()).collect(),
                signature.outputs.iter().map(|o| o.name.clone()).collect(),
            ));
        };

        // Extract execution_waves if available
        let ast_execution_waves: Vec<Vec<AstKernelCallInfo>> = if let AstNode::Program {
            execution_waves,
            ..
        } = &optimized_program
        {
            execution_waves.clone()
        } else {
            vec![]
        };

        // Compile each kernel/function with its specific config
        let shape_vars = Self::extract_shape_vars(&signature);
        let mut kernels: HashMap<String, Comp::Kernel> = HashMap::new();
        let mut kernel_call_infos: Vec<KernelCallInfo> = Vec::new();

        for func in functions {
            match func {
                AstNode::Kernel {
                    name: Some(name),
                    default_grid_size,
                    default_thread_group_size,
                    ..
                } => {
                    let grid = evaluate_dispatch_size(default_grid_size, &shape_vars);
                    let tg_base = evaluate_dispatch_size(default_thread_group_size, &shape_vars);
                    // Compute compatible local_work_size
                    let tg = [
                        Self::compute_compatible_local_size(grid[0], tg_base[0]),
                        Self::compute_compatible_local_size(grid[1], tg_base[1]),
                        Self::compute_compatible_local_size(grid[2], tg_base[2]),
                    ];

                    let mut config = KernelConfig::new(name.clone())
                        .with_global_work_size(grid)
                        .with_local_work_size(tg);

                    for (var_name, value) in &shape_vars {
                        config = config.with_shape_var(var_name.clone(), *value);
                    }

                    let kernel =
                        self.compiler
                            .compile(&self.device, &kernel_source, config.clone())?;
                    kernels.insert(name.clone(), kernel);

                    // Create KernelCallInfo for this kernel
                    kernel_call_infos.push(KernelCallInfo::new(
                        name.clone(),
                        signature.inputs.iter().map(|i| i.name.clone()).collect(),
                        signature.outputs.iter().map(|o| o.name.clone()).collect(),
                        config.global_work_size,
                        config.local_work_size.unwrap_or([1, 1, 1]),
                    ));
                }
                AstNode::Function {
                    name: Some(name), ..
                } => {
                    // Function nodes use default grid size based on output shape
                    let base_config = self.extract_kernel_config(&optimized_program, &signature);
                    let gws = base_config.global_work_size;
                    let lws_base = base_config.local_work_size.unwrap_or([1, 1, 1]);
                    let lws = [
                        Self::compute_compatible_local_size(gws[0], lws_base[0]),
                        Self::compute_compatible_local_size(gws[1], lws_base[1]),
                        Self::compute_compatible_local_size(gws[2], lws_base[2]),
                    ];

                    let config = KernelConfig::new(name.clone())
                        .with_global_work_size(gws)
                        .with_local_work_size(lws);

                    let kernel =
                        self.compiler
                            .compile(&self.device, &kernel_source, config.clone())?;
                    kernels.insert(name.clone(), kernel);

                    kernel_call_infos.push(KernelCallInfo::new(
                        name.clone(),
                        signature.inputs.iter().map(|i| i.name.clone()).collect(),
                        signature.outputs.iter().map(|o| o.name.clone()).collect(),
                        gws,
                        lws,
                    ));
                }
                _ => {}
            }
        }

        // If no kernels were found, try to compile a default kernel
        if kernels.is_empty() {
            let base_config = self.extract_kernel_config(&optimized_program, &signature);
            let gws = base_config.global_work_size;
            let lws_base = base_config.local_work_size.unwrap_or([1, 1, 1]);
            let lws = [
                Self::compute_compatible_local_size(gws[0], lws_base[0]),
                Self::compute_compatible_local_size(gws[1], lws_base[1]),
                Self::compute_compatible_local_size(gws[2], lws_base[2]),
            ];

            let config = KernelConfig::new(base_config.entry_point.clone())
                .with_global_work_size(gws)
                .with_local_work_size(lws);

            let kernel = self
                .compiler
                .compile(&self.device, &kernel_source, config)?;
            let kernel_name = base_config.entry_point.clone();
            kernels.insert(kernel_name.clone(), kernel);

            kernel_call_infos.push(KernelCallInfo::new(
                kernel_name,
                signature.inputs.iter().map(|i| i.name.clone()).collect(),
                signature.outputs.iter().map(|o| o.name.clone()).collect(),
                gws,
                lws,
            ));
        }

        // Convert to execution_waves based on ast_execution_waves
        let raw_execution_waves: Vec<Vec<KernelCallInfo>> = if !ast_execution_waves.is_empty() {
            // Use the wave structure from AST directly
            // Important: Use inputs/outputs from ast_call, not from kernel_call_infos
            ast_execution_waves
                .iter()
                .map(|wave| {
                    wave.iter()
                        .filter_map(|ast_call| {
                            // Find the compiled kernel info for dispatch sizes
                            kernel_call_infos
                                .iter()
                                .find(|k| k.kernel_name == ast_call.kernel_name)
                                .map(|k| {
                                    // Use inputs/outputs from ast_call (per-kernel info)
                                    // instead of from kernel_call_infos (global signature)
                                    KernelCallInfo::new(
                                        ast_call.kernel_name.clone(),
                                        ast_call.inputs.clone(),
                                        ast_call.outputs.clone(),
                                        k.grid_size,
                                        k.local_size,
                                    )
                                })
                        })
                        .collect()
                })
                .filter(|wave: &Vec<KernelCallInfo>| !wave.is_empty())
                .collect()
        } else {
            // Fallback: each kernel in its own wave (sequential execution)
            kernel_call_infos.into_iter().map(|k| vec![k]).collect()
        };

        // Establish data flow: chain kernel outputs to inputs with proper buffer names
        let (execution_waves, intermediate_specs) =
            Self::establish_data_flow(raw_execution_waves, &signature);

        Ok(CompiledProgram::new(
            kernels,
            execution_waves,
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
        let dispatch_config = self.extract_dispatch_config(&program, &signature);
        let kernel_config = self.extract_kernel_config(&program, &signature);
        let kernel = self
            .compiler
            .compile(&self.device, &kernel_source, kernel_config)?;

        Ok(CompiledKernel {
            kernel,
            signature,
            dispatch_config,
            _buffer: PhantomData,
        })
    }

    /// Establish data flow between kernels in execution_waves.
    ///
    /// The kernel params use placeholder names ("input0", "output") that need to be
    /// mapped to actual buffer names based on the data flow between kernels.
    ///
    /// This function:
    /// 1. Chains kernel outputs to the next kernel's inputs
    /// 2. Assigns the final kernel's output to the graph output name
    /// 3. Creates intermediate buffer names for kernels in the middle of the chain
    ///
    /// For kernels with multiple inputs (like add, mul), it tries to map inputs
    /// to the most recently produced buffers in order.
    fn establish_data_flow(
        raw_waves: Vec<Vec<KernelCallInfo>>,
        signature: &KernelSignature,
    ) -> (Vec<Vec<KernelCallInfo>>, Vec<IntermediateBufferSpec>) {
        if raw_waves.is_empty() {
            return (raw_waves, vec![]);
        }

        // Flatten waves to get kernels in execution order
        let flat_kernels: Vec<&KernelCallInfo> = raw_waves.iter().flat_map(|w| w.iter()).collect();
        let total_kernels = flat_kernels.len();

        if total_kernels == 0 {
            return (raw_waves, vec![]);
        }

        // Get external input and output names from signature
        let external_inputs: Vec<String> =
            signature.inputs.iter().map(|i| i.name.clone()).collect();
        let external_output = signature
            .outputs
            .first()
            .map(|o| o.name.clone())
            .unwrap_or_else(|| "output".to_string());

        // For each kernel, establish the data flow:
        // - First kernel reads from external inputs (if any) or nothing
        // - Each kernel writes to a buffer that later kernels may read
        // - Last kernel writes to the external output
        let mut intermediate_specs = Vec::new();
        let mut output_buffer_map: HashMap<usize, String> = HashMap::new();

        // Assign output buffer names
        for (i, _) in flat_kernels.iter().enumerate() {
            if i == total_kernels - 1 {
                // Last kernel writes to the external output
                output_buffer_map.insert(i, external_output.clone());
            } else {
                // Intermediate kernels write to temp buffers
                let temp_name = format!("__temp_{}", i);
                output_buffer_map.insert(i, temp_name.clone());

                // Create intermediate buffer spec
                // Use the output shape from signature as a reasonable default
                let shape = signature
                    .outputs
                    .first()
                    .map(|o| {
                        o.shape
                            .iter()
                            .map(|e| e.as_usize().unwrap_or(1))
                            .collect::<Vec<usize>>()
                    })
                    .unwrap_or_else(|| vec![1]);
                intermediate_specs.push(IntermediateBufferSpec::new(temp_name, shape, DType::F32));
            }
        }

        // Rebuild waves with proper buffer names
        let mut rewritten_waves = Vec::new();
        let mut kernel_idx = 0;
        // Track available buffers from previous kernels' outputs
        let mut available_buffers: Vec<String> = Vec::new();

        for wave in &raw_waves {
            let mut rewritten_wave = Vec::new();
            for call in wave {
                // Determine how many inputs this kernel expects
                let num_inputs = call.inputs.len();

                // Determine input buffer names for this kernel
                let inputs = if kernel_idx == 0 {
                    // First kernel uses external inputs, but only as many as it needs
                    if num_inputs > 0 && num_inputs < external_inputs.len() {
                        external_inputs.iter().take(num_inputs).cloned().collect()
                    } else {
                        external_inputs.clone()
                    }
                } else if num_inputs == 0 {
                    // No inputs needed
                    vec![]
                } else if num_inputs == 1 {
                    // Single input: use the most recent buffer
                    vec![available_buffers.last().cloned().unwrap_or_else(|| {
                        output_buffer_map
                            .get(&(kernel_idx - 1))
                            .cloned()
                            .unwrap_or_else(|| format!("__temp_{}", kernel_idx - 1))
                    })]
                } else {
                    // Multiple inputs: use the N most recent buffers in reverse order
                    // (most recent first for the first input, second most recent for second input, etc.)
                    let mut multi_inputs = Vec::new();
                    let available_len = available_buffers.len();
                    for i in 0..num_inputs {
                        if i < available_len {
                            // Get from available buffers (most recent first)
                            multi_inputs.push(available_buffers[available_len - 1 - i].clone());
                        } else if !external_inputs.is_empty() && i < external_inputs.len() {
                            multi_inputs.push(external_inputs[i].clone());
                        } else {
                            // Fallback: use temp buffer names
                            multi_inputs
                                .push(format!("__temp_{}", kernel_idx.saturating_sub(i + 1)));
                        }
                    }
                    // Reverse to maintain correct order (first input should be earlier buffer)
                    multi_inputs.reverse();
                    multi_inputs
                };

                // Get output buffer name
                let output_name = output_buffer_map
                    .get(&kernel_idx)
                    .cloned()
                    .unwrap_or_else(|| external_output.clone());
                let outputs = vec![output_name.clone()];

                rewritten_wave.push(KernelCallInfo::new(
                    call.kernel_name.clone(),
                    inputs,
                    outputs,
                    call.grid_size,
                    call.local_size,
                ));

                // Add this kernel's output to available buffers
                available_buffers.push(output_name);
                kernel_idx += 1;
            }
            rewritten_waves.push(rewritten_wave);
        }

        (rewritten_waves, intermediate_specs)
    }

    // Internal: optimize graph
    fn optimize_graph(&mut self, graph: Graph) -> Graph {
        use crate::opt::graph::{MultiPhaseConfig, create_multi_phase_optimizer};

        // デバイスからDeviceCapabilitiesを作成
        let opt_context = DeviceCapabilities::from_device(&self.device);

        let config = MultiPhaseConfig::new()
            .with_beam_width(self.config.graph_beam_width)
            .with_max_steps(self.config.max_steps)
            .with_progress(self.config.show_progress)
            .with_collect_logs(self.config.collect_history)
            .with_capabilities(opt_context);

        let optimizer = create_multi_phase_optimizer(config);
        let (optimized_graph, history) = optimizer.optimize_with_history(graph);

        if self.config.collect_history {
            self.histories.graph = Some(history);
        }

        optimized_graph
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
        if !simd_widths.is_empty() {
            suggesters.push(Box::new(VectorizationSuggester::with_widths(simd_widths)));
        }

        let suggester = AstCompositeSuggester::new(suggesters);
        let optimizer = AstBeamSearchOptimizer::new(suggester)
            .with_beam_width(self.config.ast_beam_width)
            .with_max_steps(self.config.max_steps)
            .with_progress(self.config.show_progress);

        let (optimized, history) = optimizer.optimize_with_history(program);

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
                    if let crate::graph::shape::Expr::Const(n) = e {
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

            let mut config = KernelConfig::new(kernel_name)
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

        KernelConfig::new("main").with_global_work_size([total_elements, 1, 1])
    }

    /// Extract shape variables from signature
    fn extract_shape_vars(signature: &KernelSignature) -> HashMap<String, i64> {
        signature.shape_vars.clone()
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
            Literal::I64(n) => *n,
            Literal::I32(n) => *n as i64,
            Literal::F32(f) => *f as i64,
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
                Literal::I64(n) => Self::Const(*n),
                Literal::I32(n) => Self::Const(*n as i64),
                Literal::F32(f) => Self::Const(*f as i64),
                Literal::Bool(b) => Self::Const(if *b { 1 } else { 0 }),
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

/// A compiled kernel with its signature
pub struct CompiledKernel<K, B>
where
    K: Kernel<Buffer = B>,
    B: Buffer,
{
    /// The compiled kernel
    pub kernel: K,
    /// The kernel signature
    pub signature: KernelSignature,
    /// Dispatch size configuration (for dynamic shape support)
    pub dispatch_config: DispatchSizeConfig,
    _buffer: PhantomData<B>,
}

/// Error type for kernel execution with query
#[derive(Debug)]
pub enum KernelExecutionError<KE> {
    /// Error during kernel execution
    KernelError(KE),
    /// Buffer not found
    BufferNotFound(String),
    /// Buffer shape mismatch
    ShapeMismatch {
        buffer_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Shape evaluation error (undefined variable, etc.)
    ShapeEvaluationError(String),
}

impl<KE: std::fmt::Display> std::fmt::Display for KernelExecutionError<KE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KernelError(e) => write!(f, "Kernel execution error: {}", e),
            Self::BufferNotFound(name) => write!(f, "Buffer not found: {}", name),
            Self::ShapeMismatch {
                buffer_name,
                expected,
                actual,
            } => write!(
                f,
                "Shape mismatch for buffer '{}': expected {:?}, got {:?}",
                buffer_name, expected, actual
            ),
            Self::ShapeEvaluationError(msg) => write!(f, "Shape evaluation error: {}", msg),
        }
    }
}

impl<KE: std::error::Error + 'static> std::error::Error for KernelExecutionError<KE> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::KernelError(e) => Some(e),
            Self::BufferNotFound(_)
            | Self::ShapeMismatch { .. }
            | Self::ShapeEvaluationError(_) => None,
        }
    }
}

impl<K, B> CompiledKernel<K, B>
where
    K: Kernel<Buffer = B>,
    B: Buffer,
{
    /// Execute the kernel with the given buffers (positional)
    pub fn execute(&self, inputs: &[&B], outputs: &mut [&mut B]) -> Result<(), K::Error> {
        self.kernel.execute(inputs, outputs)
    }

    /// Execute the kernel using an ExecutionQuery
    ///
    /// This method provides a fluent API for specifying buffers by name.
    /// The signature is used to map buffer names to the correct positions.
    /// Shape variables from the query override those from the signature for
    /// dynamic dispatch size computation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query = ExecutionQuery::new()
    ///     .input("a", &buf_a)
    ///     .input("b", &buf_b)
    ///     .output("out", &mut buf_out)
    ///     .shape_var("batch_size", 32);
    ///
    /// compiled_kernel.execute_with(query)?;
    /// ```
    pub fn execute_with(
        &self,
        mut query: ExecutionQuery<'_, B>,
    ) -> Result<(), KernelExecutionError<K::Error>> {
        // Collect input names from signature
        let input_names: Vec<String> = self
            .signature
            .inputs
            .iter()
            .map(|s| s.name.clone())
            .collect();
        let output_names: Vec<String> = self
            .signature
            .outputs
            .iter()
            .map(|s| s.name.clone())
            .collect();

        // Validate that all required buffers are present
        let missing_inputs = query.missing_inputs(&input_names);
        if !missing_inputs.is_empty() {
            return Err(KernelExecutionError::BufferNotFound(format!(
                "Missing input buffers: {:?}",
                missing_inputs
            )));
        }

        let missing_outputs = query.missing_outputs(&output_names);
        if !missing_outputs.is_empty() {
            return Err(KernelExecutionError::BufferNotFound(format!(
                "Missing output buffers: {:?}",
                missing_outputs
            )));
        }

        // Merge shape variables: signature defaults + query overrides
        let mut shape_vars = self.signature.shape_vars.clone();
        shape_vars.extend(query.get_shape_vars().iter().map(|(k, v)| (k.clone(), *v)));

        // Validate input buffer shapes
        for sig in &self.signature.inputs {
            let buffer = query.inputs().get(&sig.name).unwrap();
            let actual_shape: Vec<usize> = buffer.shape().to_vec();
            let expected_shape: Result<Vec<usize>, String> = sig
                .shape
                .iter()
                .map(|expr| expr.evaluate_usize(&shape_vars))
                .collect();
            let expected_shape =
                expected_shape.map_err(KernelExecutionError::ShapeEvaluationError)?;

            if actual_shape != expected_shape {
                return Err(KernelExecutionError::ShapeMismatch {
                    buffer_name: sig.name.clone(),
                    expected: expected_shape,
                    actual: actual_shape,
                });
            }
        }

        // Validate output buffer shapes
        for sig in &self.signature.outputs {
            // SAFETY: We already validated that all outputs exist
            let buffer = unsafe { &**query.outputs().get(&sig.name).unwrap() };
            let actual_shape: Vec<usize> = buffer.shape().to_vec();
            let expected_shape: Result<Vec<usize>, String> = sig
                .shape
                .iter()
                .map(|expr| expr.evaluate_usize(&shape_vars))
                .collect();
            let expected_shape =
                expected_shape.map_err(KernelExecutionError::ShapeEvaluationError)?;

            if actual_shape != expected_shape {
                return Err(KernelExecutionError::ShapeMismatch {
                    buffer_name: sig.name.clone(),
                    expected: expected_shape,
                    actual: actual_shape,
                });
            }
        }

        // Compute dispatch sizes using the merged shape variables
        let grid_size = self.dispatch_config.evaluate_grid_size(&shape_vars);
        let local_size = self.dispatch_config.evaluate_local_size(&shape_vars);

        // Build ordered input slice from query
        let inputs: Vec<&B> = input_names
            .iter()
            .map(|name| *query.inputs().get(name).unwrap())
            .collect();

        // Build ordered output slice from query
        // SAFETY: ExecutionQuery ensures no aliasing between outputs
        let mut outputs_map = unsafe { query.outputs_mut() };
        let mut outputs: Vec<&mut B> = output_names
            .iter()
            .map(|name| outputs_map.remove(name).unwrap())
            .collect();

        // Execute with computed sizes
        self.kernel
            .execute_with_sizes(&inputs, &mut outputs, grid_size, local_size)
            .map_err(KernelExecutionError::KernelError)
    }

    /// Get the kernel signature
    pub fn signature(&self) -> &KernelSignature {
        &self.signature
    }

    /// Get the input buffer names
    pub fn input_names(&self) -> Vec<String> {
        self.signature
            .inputs
            .iter()
            .map(|s| s.name.clone())
            .collect()
    }

    /// Get the output buffer names
    pub fn output_names(&self) -> Vec<String> {
        self.signature
            .outputs
            .iter()
            .map(|s| s.name.clone())
            .collect()
    }

    /// Create a bound execution query with default shape variables
    ///
    /// This method provides a fluent API for executing the kernel.
    /// Default shape variables from the signature are pre-populated,
    /// so you only need to specify inputs, outputs, and any overrides.
    ///
    /// # Example
    ///
    /// ```ignore
    /// compiled_kernel.query()
    ///     .input("x", &input_buf)
    ///     .output("y", &mut output_buf)
    ///     .shape_var("batch_size", 32)  // Override default
    ///     .execute()?;
    /// ```
    pub fn query(&self) -> BoundExecutionQuery<'_, K, B> {
        // Pre-populate with default shape variables from signature
        let mut query = ExecutionQuery::new();
        for (name, value) in &self.signature.shape_vars {
            query = query.shape_var(name.clone(), *value);
        }

        BoundExecutionQuery {
            kernel: self,
            query,
        }
    }
}

/// A bound execution query that holds a reference to the kernel
///
/// This struct provides a fluent API for specifying buffers and shape
/// variables, then executing the kernel in a single method chain.
///
/// # Example
///
/// ```ignore
/// compiled_kernel.query()
///     .input("x", &input_buf)
///     .output("y", &mut output_buf)
///     .execute()?;
/// ```
pub struct BoundExecutionQuery<'a, K, B>
where
    K: Kernel<Buffer = B>,
    B: Buffer,
{
    kernel: &'a CompiledKernel<K, B>,
    query: ExecutionQuery<'a, B>,
}

impl<'a, K, B> BoundExecutionQuery<'a, K, B>
where
    K: Kernel<Buffer = B>,
    B: Buffer,
{
    /// Add an input buffer with the given name
    pub fn input(mut self, name: impl Into<String>, buffer: &'a B) -> Self {
        self.query = self.query.input(name, buffer);
        self
    }

    /// Add an output buffer with the given name
    pub fn output(mut self, name: impl Into<String>, buffer: &'a mut B) -> Self {
        self.query = self.query.output(name, buffer);
        self
    }

    /// Set or override a shape variable
    ///
    /// If not called, the default value from the kernel signature is used.
    pub fn shape_var(mut self, name: impl Into<String>, value: i64) -> Self {
        self.query = self.query.shape_var(name, value);
        self
    }

    /// Set or override multiple shape variables at once
    pub fn shape_vars(mut self, vars: impl IntoIterator<Item = (String, i64)>) -> Self {
        self.query = self.query.shape_vars(vars);
        self
    }

    /// Execute the kernel with the configured buffers and shape variables
    pub fn execute(self) -> Result<(), KernelExecutionError<K::Error>> {
        self.kernel.execute_with(self.query)
    }

    /// Get the underlying ExecutionQuery (for inspection or advanced use)
    pub fn into_query(self) -> ExecutionQuery<'a, B> {
        self.query
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

    #[test]
    fn test_dispatch_size_expr_const() {
        let expr = DispatchSizeExpr::Const(42);
        let shape_vars = HashMap::new();
        assert_eq!(expr.evaluate(&shape_vars), 42);
    }

    #[test]
    fn test_dispatch_size_expr_var() {
        let expr = DispatchSizeExpr::Var("batch_size".to_string());
        let mut shape_vars = HashMap::new();
        shape_vars.insert("batch_size".to_string(), 32);

        assert_eq!(expr.evaluate(&shape_vars), 32);

        // Unknown variable defaults to 1
        let unknown = DispatchSizeExpr::Var("unknown".to_string());
        assert_eq!(unknown.evaluate(&shape_vars), 1);
    }

    #[test]
    fn test_dispatch_size_expr_arithmetic() {
        let shape_vars = HashMap::new();

        // 10 + 20 = 30
        let add = DispatchSizeExpr::Add(
            Box::new(DispatchSizeExpr::Const(10)),
            Box::new(DispatchSizeExpr::Const(20)),
        );
        assert_eq!(add.evaluate(&shape_vars), 30);

        // 10 * 5 = 50
        let mul = DispatchSizeExpr::Mul(
            Box::new(DispatchSizeExpr::Const(10)),
            Box::new(DispatchSizeExpr::Const(5)),
        );
        assert_eq!(mul.evaluate(&shape_vars), 50);

        // 100 / 10 = 10
        let div = DispatchSizeExpr::Div(
            Box::new(DispatchSizeExpr::Const(100)),
            Box::new(DispatchSizeExpr::Const(10)),
        );
        assert_eq!(div.evaluate(&shape_vars), 10);

        // max(5, 10) = 10
        let max = DispatchSizeExpr::Max(
            Box::new(DispatchSizeExpr::Const(5)),
            Box::new(DispatchSizeExpr::Const(10)),
        );
        assert_eq!(max.evaluate(&shape_vars), 10);
    }

    #[test]
    fn test_dispatch_size_expr_with_vars() {
        let mut shape_vars = HashMap::new();
        shape_vars.insert("n".to_string(), 1024);
        shape_vars.insert("block_size".to_string(), 64);

        // grid_size = n / block_size
        let expr = DispatchSizeExpr::Div(
            Box::new(DispatchSizeExpr::Var("n".to_string())),
            Box::new(DispatchSizeExpr::Var("block_size".to_string())),
        );
        assert_eq!(expr.evaluate(&shape_vars), 16);
    }

    #[test]
    fn test_dispatch_size_config_from_const() {
        let config = DispatchSizeConfig::from_const([256, 1, 1], [64, 1, 1]);
        let shape_vars = HashMap::new();

        assert_eq!(config.evaluate_grid_size(&shape_vars), [256, 1, 1]);
        assert_eq!(config.evaluate_local_size(&shape_vars), [64, 1, 1]);
    }

    #[test]
    fn test_dispatch_size_config_dynamic() {
        // Create config with variable-based expressions
        let config = DispatchSizeConfig {
            grid_size: [
                DispatchSizeExpr::Var("total_elements".to_string()),
                DispatchSizeExpr::Const(1),
                DispatchSizeExpr::Const(1),
            ],
            local_size: [
                DispatchSizeExpr::Const(64),
                DispatchSizeExpr::Const(1),
                DispatchSizeExpr::Const(1),
            ],
        };

        // Different shape_vars produce different sizes
        let mut vars1 = HashMap::new();
        vars1.insert("total_elements".to_string(), 1024);
        assert_eq!(config.evaluate_grid_size(&vars1), [1024, 1, 1]);

        let mut vars2 = HashMap::new();
        vars2.insert("total_elements".to_string(), 2048);
        assert_eq!(config.evaluate_grid_size(&vars2), [2048, 1, 1]);
    }

    #[test]
    fn test_dispatch_size_expr_from_ast() {
        // Test conversion from AstNode
        let ast = AstNode::Mul(
            Box::new(AstNode::Var("n".to_string())),
            Box::new(const_int(2)),
        );

        let expr = DispatchSizeExpr::from_ast(&ast);

        let mut shape_vars = HashMap::new();
        shape_vars.insert("n".to_string(), 100);

        assert_eq!(expr.evaluate(&shape_vars), 200);
    }
}
