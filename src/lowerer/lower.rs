//! Main lowering logic for converting GraphNode to AST
//!
//! This module implements the core lowering algorithm that converts
//! computation graph nodes into executable AST kernels.

use std::collections::HashMap;
use std::fmt;

use crate::ast::{AddressSpace, AstNode, DType, Literal, Mutability, VarDecl, VarKind};
use crate::graph::{Expr, GraphNode, GraphOp, ReduceOp, View, collect_inputs, topological_sort};

use super::fusion::AllFusions;
use super::fusion::FusionPass;
use super::index_gen::IndexGenerator;
use super::loop_gen::LoopGenerator;

/// Errors that can occur during lowering
#[derive(Debug, Clone)]
pub enum LoweringError {
    /// Buffer not found for a graph node
    BufferNotFound { node_info: String },
    /// Invalid wildcard ID in expression
    InvalidWildcardId { id: String },
    /// Unsupported operation type
    UnsupportedOperation { op_name: String },
    /// Shape mismatch during lowering
    ShapeMismatch { expected: String, actual: String },
    /// Internal error (unexpected state)
    InternalError { message: String },
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoweringError::BufferNotFound { node_info } => {
                write!(f, "Buffer not found for node: {}", node_info)
            }
            LoweringError::InvalidWildcardId { id } => {
                write!(f, "Invalid wildcard ID: {}", id)
            }
            LoweringError::UnsupportedOperation { op_name } => {
                write!(f, "Unsupported operation: {}", op_name)
            }
            LoweringError::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {}, got {}", expected, actual)
            }
            LoweringError::InternalError { message } => {
                write!(f, "Internal error: {}", message)
            }
        }
    }
}

impl std::error::Error for LoweringError {}

/// Result type for lowering operations
pub type LoweringResult<T> = Result<T, LoweringError>;

/// Buffer information for kernel parameters
#[derive(Clone, Debug)]
struct BufferInfo {
    name: String,
    dtype: DType,
    shape: Vec<crate::graph::Expr>,
}

/// Context for kernel generation (reduces code duplication)
struct KernelContext {
    kernel_name: String,
    input_buffers: Vec<BufferInfo>,
    output_buffer: BufferInfo,
}

/// Kernel operation type for naming (tinygrad-style)
#[derive(Clone, Copy, Debug)]
enum KernelOpType {
    /// Elementwise operation
    Elementwise,
    /// Reduce operation
    Reduce,
    /// Copy/View operation
    Copy,
    /// Scatter operation (fold)
    Scatter,
    /// Cumulative scan operation
    Scan,
}

impl KernelOpType {
    /// Get the prefix character for kernel naming
    fn prefix(&self) -> &'static str {
        match self {
            KernelOpType::Elementwise => "E",
            KernelOpType::Reduce => "R",
            KernelOpType::Copy => "C",
            KernelOpType::Scatter => "S",
            KernelOpType::Scan => "P", // P for Prefix scan
        }
    }
}

/// Main lowerer for converting computation graphs to AST
pub struct Lowerer {
    /// Counter for generating unique buffer names
    buffer_counter: usize,
    /// Counter for generating unique kernel names
    kernel_counter: usize,
    /// Mapping from graph nodes to buffer names
    buffer_map: HashMap<*const crate::graph::GraphInner, String>,
    /// Mapping from graph nodes to dtypes
    dtype_map: HashMap<*const crate::graph::GraphInner, DType>,
    /// Mapping from graph nodes to shapes
    shape_map: HashMap<*const crate::graph::GraphInner, Vec<crate::graph::Expr>>,
    /// Generated kernels
    kernels: Vec<AstNode>,
    /// Loop generator
    loop_gen: LoopGenerator,
    /// Consumer count for each node (used for View inlining decisions)
    consumer_counts: HashMap<*const crate::graph::GraphInner, usize>,
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowerer {
    /// Create a new lowerer
    pub fn new() -> Self {
        Self {
            buffer_counter: 0,
            kernel_counter: 0,
            buffer_map: HashMap::new(),
            dtype_map: HashMap::new(),
            shape_map: HashMap::new(),
            kernels: Vec::new(),
            loop_gen: LoopGenerator::new(),
            consumer_counts: HashMap::new(),
        }
    }

    /// Get the index generator
    pub fn index_gen(&self) -> &IndexGenerator {
        self.loop_gen.index_gen()
    }

    // =========================================================================
    // View inlining support
    // =========================================================================

    /// Count consumers for each node in the graph
    ///
    /// Used to determine if a View can be inlined (single consumer = safe to inline)
    fn count_consumers(nodes: &[GraphNode]) -> HashMap<*const crate::graph::GraphInner, usize> {
        use std::collections::HashSet;

        let mut counts: HashMap<*const crate::graph::GraphInner, usize> = HashMap::new();
        let mut visited = HashSet::new();

        fn traverse(
            node: &GraphNode,
            counts: &mut HashMap<*const crate::graph::GraphInner, usize>,
            visited: &mut HashSet<*const crate::graph::GraphInner>,
        ) {
            let ptr = std::rc::Rc::as_ptr(&node.0);
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in node.sources() {
                let src_ptr = std::rc::Rc::as_ptr(&src.0);
                *counts.entry(src_ptr).or_insert(0) += 1;
                traverse(&src, counts, visited);
            }
        }

        for node in nodes {
            traverse(node, &mut counts, &mut visited);
        }

        counts
    }

    /// Get the consumer count for a node
    fn consumer_count(&self, node: &GraphNode) -> usize {
        let ptr = std::rc::Rc::as_ptr(&node.0);
        self.consumer_counts.get(&ptr).copied().unwrap_or(0)
    }

    /// Check if a View node can be inlined into its consumer
    ///
    /// A View can be inlined if:
    /// 1. It has at most one consumer
    /// 2. It has sources (not an external input)
    /// Multiple consumers require a buffer to avoid redundant computation.
    fn can_inline_view(&self, view_node: &GraphNode) -> bool {
        // External inputs or nodes without sources cannot be inlined
        if view_node.is_external() || view_node.sources().is_empty() {
            return false;
        }
        self.consumer_count(view_node) <= 1
    }

    /// Resolve a chain of View operations to find the actual data source
    ///
    /// Returns (actual_source, composed_view) where:
    /// - actual_source: The first non-View node (or View that can't be inlined)
    /// - composed_view: The combined View transformation from actual_source to the original node
    fn resolve_view_chain(&self, node: &GraphNode) -> (GraphNode, View) {
        match node.op() {
            GraphOp::View(source_view) if self.can_inline_view(node) => {
                // View can be inlined: recursively resolve source
                let (inner_src, inner_view) = self.resolve_view_chain(&node.sources()[0]);
                // Compose: node's view applied to inner_view
                let composed = View::compose(node.view(), &inner_view);

                (inner_src, composed)
            }
            _ => {
                // Not a View or can't be inlined: this is our actual source
                (node.clone(), node.view().clone())
            }
        }
    }

    /// Generate a new unique buffer name
    fn next_buffer_name(&mut self) -> String {
        let name = format!("buf{}", self.buffer_counter);
        self.buffer_counter += 1;
        name
    }

    /// Generate kernel name in tinygrad-style: {op_type}_{shape}_{counter}
    ///
    /// Examples:
    /// - Elementwise [256, 256] → "E_256_256_0"
    /// - Reduce [32, 64] → "R_32_64_1"
    /// - Copy [128] → "C_128_2"
    fn generate_kernel_name(&mut self, op_type: KernelOpType, node: &GraphNode) -> String {
        let prefix = op_type.prefix();
        let shape = node.shape();

        // Format shape as _dim1_dim2_...
        let shape_str: String = shape
            .iter()
            .map(|dim| {
                // Try to get constant value, otherwise use "N" for dynamic
                dim.as_const().map_or("N".to_string(), |v| v.to_string())
            })
            .collect::<Vec<_>>()
            .join("_");

        let name = format!("{}_{}", prefix, shape_str);
        let kernel_id = self.kernel_counter;
        self.kernel_counter += 1;

        format!("{}_{}", name, kernel_id)
    }

    /// Get or create buffer name for a node
    fn get_buffer_name(&mut self, node: &GraphNode) -> String {
        let ptr = std::rc::Rc::as_ptr(&node.0);
        if let Some(name) = self.buffer_map.get(&ptr) {
            return name.clone();
        }

        // Check if node has a user-provided name
        let name = if let Some(user_name) = node.name() {
            format!("{}_{}", user_name, self.buffer_counter)
        } else {
            self.next_buffer_name()
        };
        self.buffer_counter += 1;

        self.buffer_map.insert(ptr, name.clone());
        self.dtype_map.insert(ptr, node.dtype().clone());
        self.shape_map.insert(ptr, node.shape().clone());
        name
    }

    /// Get buffer dtype for a node
    fn get_buffer_dtype(&self, node: &GraphNode) -> DType {
        let ptr = std::rc::Rc::as_ptr(&node.0);
        self.dtype_map.get(&ptr).cloned().unwrap_or(DType::F32)
    }

    /// Lower a computation graph to a Program AST
    ///
    /// This is the main entry point. It:
    /// 1. Applies fusion passes
    /// 2. Topologically sorts nodes
    /// 3. Generates a kernel for each non-input node
    /// 4. Wraps everything in a Program node
    ///
    /// # Errors
    ///
    /// Returns `LoweringError` if:
    /// - An unsupported operation is encountered
    /// - Buffer lookup fails
    /// - Shape validation fails
    pub fn lower(&mut self, roots: &[GraphNode]) -> LoweringResult<AstNode> {
        // Apply fusion passes
        let fused = AllFusions.apply(roots);

        // Get topological order
        let sorted = topological_sort(&fused);

        // Compute consumer counts for View inlining decisions
        self.consumer_counts = Self::count_consumers(&sorted);

        // Identify external inputs
        let inputs = collect_inputs(&fused);
        for input in &inputs {
            self.get_buffer_name(input);
        }

        // Generate kernels for each non-input node
        for node in sorted {
            if !node.is_external() {
                if let Some(kernel) = self.lower_node(&node)? {
                    self.kernels.push(kernel);
                }
            }
        }

        // Create Program node
        Ok(AstNode::Program {
            functions: self.kernels.clone(),
            execution_waves: vec![], // TODO: dependency analysis
        })
    }

    /// Lower a single graph node to a Kernel AST
    ///
    /// Returns `Ok(None)` for View nodes that can be inlined into consumers.
    fn lower_node(&mut self, node: &GraphNode) -> LoweringResult<Option<AstNode>> {
        // Check if this View can be inlined (skip kernel generation)
        if matches!(node.op(), GraphOp::View(_)) && self.can_inline_view(node) {
            return Ok(None);
        }

        let output_buf = self.get_buffer_name(node);

        let result = match node.op() {
            GraphOp::View(_view) => {
                // View operation that cannot be inlined: generate copy kernel
                self.lower_view(node, &output_buf)
            }
            GraphOp::MapReduce { map, reduce } => {
                if let Some((reduce_op, axis)) = reduce {
                    self.lower_reduce(node, &output_buf, map, *reduce_op, *axis)
                } else {
                    self.lower_elementwise(node, &output_buf, map)
                }
            }
            GraphOp::Unfold { .. } => {
                // Unfold is implemented as a view transformation
                // The view already contains the correct index expression
                self.lower_view(node, &output_buf)
            }
            GraphOp::Scatter {
                output_shape,
                axes,
                sizes,
                strides,
                dilations,
            } => self.lower_scatter(
                node,
                &output_buf,
                output_shape,
                axes,
                sizes,
                strides,
                dilations,
            ),
            GraphOp::Scan {
                map,
                scan_op,
                axis,
                exclusive,
            } => self.lower_scan(node, &output_buf, map, *scan_op, *axis, *exclusive),
        };

        Ok(Some(result))
    }

    /// Collect buffer info for input sources (deduplicated)
    ///
    /// Resolves View chains to use the actual source buffers.
    fn collect_input_buffers(&mut self, node: &GraphNode) -> Vec<BufferInfo> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for src in node.sources() {
            // Resolve View chain to get actual source
            let (actual_src, _) = self.resolve_view_chain(&src);
            let name = self.get_buffer_name(&actual_src);

            // Only include if we haven't seen this buffer name before
            if seen.insert(name.clone()) {
                result.push(BufferInfo {
                    name,
                    dtype: self.get_buffer_dtype(&actual_src),
                    shape: actual_src.shape().clone(),
                });
            }
        }

        result
    }

    /// Prepare kernel context (common setup for all kernel-lowering methods)
    fn prepare_kernel_context(
        &mut self,
        node: &GraphNode,
        output_buf: &str,
        op_type: KernelOpType,
        output_shape: Option<&[Expr]>,
    ) -> KernelContext {
        let input_buffers = self.collect_input_buffers(node);
        let output_buffer = BufferInfo {
            name: output_buf.to_string(),
            dtype: node.dtype().clone(),
            shape: output_shape
                .map(|s| s.to_vec())
                .unwrap_or_else(|| node.shape().clone()),
        };
        let kernel_name = self.generate_kernel_name(op_type, node);
        KernelContext {
            kernel_name,
            input_buffers,
            output_buffer,
        }
    }

    /// Finalize kernel by wrapping body with make_kernel
    fn finalize_kernel(&mut self, ctx: KernelContext, body: AstNode) -> AstNode {
        self.make_kernel(&ctx.kernel_name, body, ctx.input_buffers, ctx.output_buffer)
    }

    /// Lower a View operation
    ///
    /// When the view has bounds (e.g., for padding), the load is wrapped in a
    /// Select that returns the default value when out of bounds.
    fn lower_view(&mut self, node: &GraphNode, output_buf: &str) -> AstNode {
        let ctx = self.prepare_kernel_context(node, output_buf, KernelOpType::Copy, None);

        let shape = node.shape();
        let output_idx = self.index_gen().view_to_index(node.view());

        // Resolve View chain to get actual source and composed view
        let src = &node.sources()[0];
        let (actual_src, view_for_indexing) = self.resolve_view_chain(src);

        let src_buf = self.get_buffer_name(&actual_src);
        let src_idx = self.index_gen().view_to_index(&view_for_indexing);

        // Load from source
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Var(src_buf)),
            offset: Box::new(src_idx),
            count: 1,
            dtype: node.dtype().clone(),
        };

        // Handle bounds (for padded views)
        let load_expr = if let Some(default_value) = view_for_indexing.default_value() {
            // View has bounds - wrap load in conditional
            let condition = view_for_indexing.condition().unwrap();
            let cond_ast: AstNode = condition.clone().into();
            let default_ast = AstNode::Const(crate::ast::Literal::F32(default_value.as_f32()));
            AstNode::Select {
                cond: Box::new(cond_ast),
                then_val: Box::new(load),
                else_val: Box::new(default_ast),
            }
        } else {
            load
        };

        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(output_idx),
            value: Box::new(load_expr),
        };

        let body = self.loop_gen.generate_loops(&shape, store);

        self.finalize_kernel(ctx, body)
    }

    /// Lower an elementwise (MapReduce with reduce=None) operation
    fn lower_elementwise(&mut self, node: &GraphNode, output_buf: &str, map: &AstNode) -> AstNode {
        let ctx = self.prepare_kernel_context(node, output_buf, KernelOpType::Elementwise, None);

        let shape = node.shape();
        let output_idx = self.index_gen().view_to_index(node.view());

        // Substitute Wildcards with Load operations
        let element_expr = self.substitute_wildcards(node, map);

        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(output_idx),
            value: Box::new(element_expr),
        };

        let body = self.loop_gen.generate_loops(&shape, store);

        self.finalize_kernel(ctx, body)
    }

    /// Lower a reduction operation
    fn lower_reduce(
        &mut self,
        node: &GraphNode,
        output_buf: &str,
        map: &AstNode,
        reduce_op: ReduceOp,
        axis: usize,
    ) -> AstNode {
        let ctx = self.prepare_kernel_context(node, output_buf, KernelOpType::Reduce, None);

        // Use source shape for iteration (before reduction)
        let src = &node.sources()[0];
        let src_shape = src.shape();

        // Output index for reduce: substitute the reduce axis index with 0
        // since the output dimension at the reduce axis has size 1
        let output_idx = self
            .index_gen()
            .view_to_reduce_output_index(node.view(), axis);

        // Identity value
        let identity = self.reduce_identity(&reduce_op, node.dtype());

        // Substitute Wildcards in map expression
        let load_expr = self.substitute_wildcards(node, map);

        // Combine expression: acc = reduce_op(acc, val)
        let acc_var = AstNode::Var("acc".to_string());
        let combine_expr = reduce_op.combine(acc_var, load_expr);

        // Get the output dtype for the accumulator
        let acc_dtype = node.dtype();

        // Generate reduce loop structure
        let body = self.loop_gen.generate_reduce(
            &src_shape,
            axis,
            AstNode::Var(output_buf.to_string()),
            output_idx,
            "acc",
            acc_dtype,
            identity,
            combine_expr,
        );

        self.finalize_kernel(ctx, body)
    }

    /// Lower a Scatter operation (fold)
    ///
    /// Scatter-add accumulates values from unfolded windows back to the original shape.
    /// This is implemented in two steps:
    /// 1. Initialize output buffer to zero
    /// 2. Iterate over input (unfolded) shape and atomic-add to output positions
    #[allow(clippy::too_many_arguments)]
    fn lower_scatter(
        &mut self,
        node: &GraphNode,
        output_buf: &str,
        output_shape: &[Expr],
        axes: &[usize],
        _sizes: &[Expr],
        strides: &[Expr],
        dilations: &[Expr],
    ) -> AstNode {
        use crate::ast::Scope;

        let ctx = self.prepare_kernel_context(
            node,
            output_buf,
            KernelOpType::Scatter,
            Some(output_shape),
        );

        let src = &node.sources()[0];
        let src_shape = src.shape();
        let dtype = node.dtype().clone();

        // --- Step 1: Initialize output to zero ---
        let output_view = View::contiguous(output_shape.to_vec());
        let output_idx = self.index_gen().view_to_index(&output_view);
        let zero = AstNode::Const(Literal::F32(0.0));
        let zero_store = AstNode::Store {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(output_idx),
            value: Box::new(zero),
        };
        let init_loop = self.loop_gen.generate_loops(output_shape, zero_store);

        // --- Step 2: Scatter-add from input to output ---
        // Input shape: [preserved_dims..., out_positions..., window_dims...]
        // We need to map each input element to its corresponding output position

        let output_ndim = output_shape.len();
        let num_unfolded = axes.len();
        let num_preserved = output_ndim - num_unfolded;

        // Load from source
        let src_buf = self.get_buffer_name(src);
        let src_idx = self.index_gen().view_to_index(src.view());
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Var(src_buf)),
            offset: Box::new(src_idx),
            count: 1,
            dtype: dtype.clone(),
        };

        // Compute destination index
        // For each element in the unfolded tensor, we compute where it came from in the original
        // input_idx[axis] = out_pos * stride + win_pos * dilation
        let mut dst_idx = Expr::Const(0);
        let mut dst_stride = Expr::Const(1);

        // Process axes in reverse order for proper stride calculation
        for axis in (0..output_ndim).rev() {
            let axis_size = &output_shape[axis];

            let coord = if let Some(unfold_i) = axes.iter().position(|&a| a == axis) {
                // This axis was unfolded
                // out_pos is at index (num_preserved + unfold_i) in src_shape
                // win_pos is at index (num_preserved + num_unfolded + unfold_i) in src_shape
                let out_pos_idx = num_preserved + unfold_i;
                let win_pos_idx = num_preserved + num_unfolded + unfold_i;
                let unfold_stride = &strides[unfold_i];
                let dilation = &dilations[unfold_i];

                // coord = out_pos * stride + win_pos * dilation
                (Expr::Idx(out_pos_idx) * unfold_stride.clone()
                    + Expr::Idx(win_pos_idx) * dilation.clone())
                .simplify()
            } else {
                // Preserved axis - find its position in the src_shape
                let preserved_idx = (0..axis).filter(|a| !axes.contains(a)).count();
                Expr::Idx(preserved_idx)
            };

            dst_idx = (dst_idx + coord * dst_stride.clone()).simplify();
            dst_stride = (dst_stride * axis_size.clone()).simplify();
        }

        // Atomic add to output
        let dst_idx_ast: AstNode = dst_idx.into();
        let atomic_add = AstNode::AtomicAdd {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(dst_idx_ast),
            value: Box::new(load),
            dtype,
        };

        let scatter_loop = self.loop_gen.generate_loops(&src_shape, atomic_add);

        // Combine init and scatter loops
        let body = AstNode::Block {
            statements: vec![init_loop, scatter_loop],
            scope: Box::new(Scope::new()),
        };

        self.finalize_kernel(ctx, body)
    }

    /// Lower a Scan (cumulative) operation
    ///
    /// Generates a sequential scan kernel that computes cumulative values along an axis.
    /// For cumsum: output[i] = sum(input[0..=i])
    /// The output has the same shape as the input.
    #[allow(clippy::too_many_arguments)]
    fn lower_scan(
        &mut self,
        node: &GraphNode,
        output_buf: &str,
        map: &AstNode,
        scan_op: ReduceOp,
        axis: usize,
        _exclusive: bool, // Reserved for future use
    ) -> AstNode {
        let ctx = self.prepare_kernel_context(node, output_buf, KernelOpType::Scan, None);

        // Use source shape for iteration (output shape is the same)
        let src = &node.sources()[0];
        let src_shape = src.shape();

        // Output index: same indexing as input (shape is preserved)
        let output_idx = self.index_gen().view_to_index(node.view());

        // Identity value for the scan operation
        let identity = self.reduce_identity(&scan_op, node.dtype());

        // Substitute Wildcards in map expression
        let load_expr = self.substitute_wildcards(node, map);

        // Combine expression: acc = scan_op(acc, val)
        let acc_var = AstNode::Var("acc".to_string());
        let combine_expr = scan_op.combine(acc_var, load_expr);

        // Get the output dtype for the accumulator
        let acc_dtype = node.dtype();

        // Generate scan loop structure
        let body = self.loop_gen.generate_scan(
            &src_shape,
            axis,
            AstNode::Var(output_buf.to_string()),
            output_idx,
            "acc",
            acc_dtype,
            identity,
            combine_expr,
        );

        self.finalize_kernel(ctx, body)
    }

    /// Collect symbolic shape variables from shapes
    ///
    /// Extracts all `Expr::Sym` variables from the given shapes and returns
    /// them as a sorted, deduplicated list.
    fn collect_shape_vars(shapes: &[&[Expr]]) -> Vec<String> {
        use std::collections::HashSet;

        fn collect_from_expr(expr: &Expr, vars: &mut HashSet<String>) {
            match expr {
                Expr::Sym(name) => {
                    vars.insert(name.clone());
                }
                Expr::Add(l, r)
                | Expr::Sub(l, r)
                | Expr::Mul(l, r)
                | Expr::Div(l, r)
                | Expr::Rem(l, r)
                | Expr::Lt(l, r)
                | Expr::And(l, r) => {
                    collect_from_expr(l, vars);
                    collect_from_expr(r, vars);
                }
                Expr::Not(a) => {
                    collect_from_expr(a, vars);
                }
                Expr::LoadIndex { offset_expr, .. } => {
                    collect_from_expr(offset_expr, vars);
                }
                Expr::Const(_) | Expr::Bool(_) | Expr::Idx(_) => {}
            }
        }

        let mut vars = HashSet::new();
        for shape in shapes {
            for expr in *shape {
                collect_from_expr(expr, &mut vars);
            }
        }

        let mut result: Vec<_> = vars.into_iter().collect();
        result.sort(); // Ensure deterministic order
        result
    }

    /// Create a Kernel AST node with buffer parameters
    fn make_kernel(
        &self,
        name: &str,
        body: AstNode,
        input_buffers: Vec<BufferInfo>,
        output_buffer: BufferInfo,
    ) -> AstNode {
        let one = Box::new(AstNode::Const(Literal::I64(1)));

        // Compute grid size from output shape
        // Grid size is the total number of elements to process
        let grid_size = self.shape_to_grid_size(&output_buffer.shape);

        // Collect shape variables from all buffer shapes
        let all_shapes: Vec<&[Expr]> = input_buffers
            .iter()
            .map(|b| b.shape.as_slice())
            .chain(std::iter::once(output_buffer.shape.as_slice()))
            .collect();
        let shape_vars = Self::collect_shape_vars(&all_shapes);

        // Create parameters for input buffers
        let mut params: Vec<VarDecl> = input_buffers
            .into_iter()
            .map(|buf| VarDecl {
                name: buf.name,
                dtype: DType::Ptr(Box::new(buf.dtype), AddressSpace::Global),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            })
            .collect();

        // Add output buffer parameter
        params.push(VarDecl {
            name: output_buffer.name,
            dtype: DType::Ptr(Box::new(output_buffer.dtype), AddressSpace::Global),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

        // Add shape variable parameters (after buffer parameters)
        for var_name in shape_vars {
            params.push(VarDecl {
                name: var_name,
                dtype: DType::I64,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
        }

        AstNode::Kernel {
            name: Some(name.to_string()),
            params,
            return_type: DType::Void,
            body: Box::new(body),
            default_grid_size: grid_size,
            default_thread_group_size: [one.clone(), one.clone(), one],
        }
    }

    /// Convert shape to grid size for kernel dispatch
    ///
    /// Currently returns [1, 1, 1] to use sequential loop execution.
    /// The kernel body contains Range loops that iterate over all elements.
    /// TODO: For parallel execution, replace Range loops with thread indexing
    /// and set grid_size to the actual output shape.
    fn shape_to_grid_size(&self, _shape: &[crate::graph::Expr]) -> [Box<AstNode>; 3] {
        let one = Box::new(AstNode::Const(Literal::I64(1)));
        // Use sequential execution for now - a single thread runs all loops
        [one.clone(), one.clone(), one]
    }

    /// Substitute Wildcard nodes with Load operations
    ///
    /// Creates a mapping from wildcard IDs ("0", "1", ...) to Load operations
    /// for each source. Resolves View chains to load directly from the actual
    /// source with composed index expressions.
    ///
    /// When a view has bounds (e.g., for padding), the load is wrapped in a
    /// Select that returns the default value when out of bounds.
    fn substitute_wildcards(&mut self, node: &GraphNode, expr: &AstNode) -> AstNode {
        let mut mappings = std::collections::HashMap::new();

        for (i, src) in node.sources().iter().enumerate() {
            // Resolve View chain to get actual source and composed view
            let (actual_src, view_for_indexing) = self.resolve_view_chain(src);

            let src_buf = self.get_buffer_name(&actual_src);
            let idx = self.index_gen().view_to_index(&view_for_indexing);

            let load = AstNode::Load {
                ptr: Box::new(AstNode::Var(src_buf)),
                offset: Box::new(idx),
                count: 1,
                dtype: src.dtype().clone(),
            };

            // Handle bounds (for padded views)
            let load_expr = if let Some(default_value) = view_for_indexing.default_value() {
                // View has bounds - wrap load in conditional
                let condition = view_for_indexing.condition().unwrap();
                let cond_ast: AstNode = condition.clone().into();
                let default_ast = AstNode::Const(crate::ast::Literal::F32(default_value.as_f32()));
                AstNode::Select {
                    cond: Box::new(cond_ast),
                    then_val: Box::new(load),
                    else_val: Box::new(default_ast),
                }
            } else {
                load
            };

            mappings.insert(i.to_string(), load_expr);
        }

        expr.substitute(&mappings)
    }

    /// Get the identity value for a reduction operation
    fn reduce_identity(&self, op: &ReduceOp, dtype: &DType) -> AstNode {
        let lit = op.identity(dtype);
        AstNode::Const(lit)
    }

    // =========================================================================
    // Public API for buffer map access
    // =========================================================================

    /// Get a reference to the buffer map
    ///
    /// The buffer map contains mappings from GraphNode raw pointers to buffer names.
    /// This is useful for retrieving buffer names after lowering.
    pub fn buffer_map(&self) -> &HashMap<*const crate::graph::GraphInner, String> {
        &self.buffer_map
    }

    /// Get a reference to the dtype map
    ///
    /// The dtype map contains mappings from GraphNode raw pointers to DTypes.
    pub fn dtype_map(&self) -> &HashMap<*const crate::graph::GraphInner, DType> {
        &self.dtype_map
    }

    /// Look up the buffer name for a given graph node
    ///
    /// Returns `None` if the node has not been assigned a buffer.
    /// Unlike the internal `get_buffer_name`, this does not create new buffers.
    pub fn lookup_buffer_name(&self, node: &GraphNode) -> Option<&String> {
        let ptr = std::rc::Rc::as_ptr(&node.0);
        self.buffer_map.get(&ptr)
    }

    /// Look up the dtype for a given graph node
    ///
    /// Returns `None` if the node's dtype has not been recorded.
    pub fn lookup_dtype(&self, node: &GraphNode) -> Option<DType> {
        let ptr = std::rc::Rc::as_ptr(&node.0);
        self.dtype_map.get(&ptr).cloned()
    }

    /// Get buffer information by name
    ///
    /// Returns (shape, dtype) for the given buffer name.
    /// This is useful for allocating intermediate buffers during execution.
    pub fn get_buffer_info_by_name(&self, name: &str) -> Option<(Vec<crate::graph::Expr>, DType)> {
        // Find the node with this buffer name
        for (ptr, buf_name) in &self.buffer_map {
            if buf_name == name {
                let dtype = self.dtype_map.get(ptr).cloned().unwrap_or(DType::F32);
                let shape = self.shape_map.get(ptr).cloned().unwrap_or_default();
                return Some((shape, dtype));
            }
        }
        None
    }

    /// Get all buffer names with their shapes and dtypes
    ///
    /// Returns a map from buffer name to (shape, dtype).
    pub fn get_all_buffer_info(&self) -> HashMap<String, (Vec<crate::graph::Expr>, DType)> {
        let mut result = HashMap::new();
        for (ptr, name) in &self.buffer_map {
            let dtype = self.dtype_map.get(ptr).cloned().unwrap_or(DType::F32);
            let shape = self.shape_map.get(ptr).cloned().unwrap_or_default();
            result.insert(name.clone(), (shape, dtype));
        }
        result
    }

    /// Build a KernelSignature from input and output nodes
    ///
    /// This function creates a signature containing buffer information
    /// for the specified input and output nodes.
    pub fn build_kernel_signature(
        &self,
        inputs: &[GraphNode],
        outputs: &[GraphNode],
    ) -> crate::backend::KernelSignature {
        use crate::backend::{BufferSignature, KernelSignature};
        use crate::shape::Expr;

        let build_buffer_sig = |node: &GraphNode| -> BufferSignature {
            let name = self
                .lookup_buffer_name(node)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            let shape: Vec<Expr> = node.shape().clone();
            BufferSignature::new(name, shape)
        };

        let input_sigs: Vec<BufferSignature> = inputs.iter().map(build_buffer_sig).collect();
        let output_sigs: Vec<BufferSignature> = outputs.iter().map(build_buffer_sig).collect();

        KernelSignature::new(input_sigs, output_sigs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Expr, input};

    #[test]
    fn test_lower_simple_elementwise() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("x");
        let y = (&x + &x).with_name("y");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[y]).expect("Lowering should succeed");

        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_lower_reduction() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("x");
        let y = x.sum(1).with_name("y");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[y]).expect("Lowering should succeed");

        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_lower_complex_graph() {
        let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
        let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");

        // (a + b).sum(1)
        let c = (&a + &b).sum(1).with_name("c");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[c]).expect("Lowering should succeed");

        assert!(matches!(program, AstNode::Program { functions, .. } if !functions.is_empty()));
    }

    #[test]
    fn test_collect_shape_vars() {
        // Test with no shape vars
        let shapes: Vec<&[Expr]> = vec![&[Expr::Const(32), Expr::Const(64)]];
        let vars = Lowerer::collect_shape_vars(&shapes);
        assert!(vars.is_empty());

        // Test with single shape var
        let shape1 = [Expr::Sym("batch".to_string()), Expr::Const(64)];
        let shapes: Vec<&[Expr]> = vec![shape1.as_slice()];
        let vars = Lowerer::collect_shape_vars(&shapes);
        assert_eq!(vars, vec!["batch".to_string()]);

        // Test with multiple shape vars (should be sorted)
        let shape2 = [
            Expr::Sym("batch".to_string()),
            Expr::Sym("seq_len".to_string()),
        ];
        let shapes: Vec<&[Expr]> = vec![shape2.as_slice()];
        let vars = Lowerer::collect_shape_vars(&shapes);
        assert_eq!(vars, vec!["batch".to_string(), "seq_len".to_string()]);

        // Test deduplication across multiple shapes
        let shape_a = [Expr::Sym("batch".to_string()), Expr::Const(64)];
        let shape_b = [Expr::Sym("batch".to_string()), Expr::Const(128)];
        let shapes: Vec<&[Expr]> = vec![shape_a.as_slice(), shape_b.as_slice()];
        let vars = Lowerer::collect_shape_vars(&shapes);
        assert_eq!(vars, vec!["batch".to_string()]);
    }

    #[test]
    fn test_lower_dynamic_shape_elementwise() {
        // Create tensor with dynamic batch dimension
        let batch = Expr::Sym("batch".to_string());
        let x = input(vec![batch.clone(), Expr::Const(64)], DType::F32).with_name("x");
        let y = (&x + &x).with_name("y");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[y]).expect("Lowering should succeed");

        // Verify program was generated
        assert!(matches!(program, AstNode::Program { .. }));

        // Extract kernel and check for shape variable parameter
        if let AstNode::Program { functions, .. } = program {
            assert!(!functions.is_empty());
            let kernel = &functions[0];
            if let AstNode::Kernel { params, .. } = kernel {
                // Should have: input buffer, output buffer, and "batch" shape var
                let batch_param = params.iter().find(|p| p.name == "batch");
                assert!(
                    batch_param.is_some(),
                    "Expected 'batch' shape variable parameter"
                );
                assert_eq!(batch_param.unwrap().dtype, DType::I64);
            }
        }
    }

    #[test]
    fn test_lower_dynamic_shape_reduction() {
        // Create tensor with dynamic shape
        let batch = Expr::Sym("batch".to_string());
        let seq_len = Expr::Sym("seq_len".to_string());
        let x = input(vec![batch.clone(), seq_len.clone()], DType::F32).with_name("x");
        let y = x.sum(1).with_name("y");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[y]).expect("Lowering should succeed");

        // Verify program was generated
        assert!(matches!(program, AstNode::Program { .. }));

        // Extract kernel and check for shape variable parameters
        if let AstNode::Program { functions, .. } = program {
            assert!(!functions.is_empty());
            let kernel = &functions[0];
            if let AstNode::Kernel { params, .. } = kernel {
                // Should have shape variable parameters (sorted alphabetically)
                let shape_params: Vec<_> = params
                    .iter()
                    .filter(|p| p.dtype == DType::I64)
                    .map(|p| p.name.clone())
                    .collect();
                assert!(
                    shape_params.contains(&"batch".to_string()),
                    "Expected 'batch' shape variable"
                );
                assert!(
                    shape_params.contains(&"seq_len".to_string()),
                    "Expected 'seq_len' shape variable"
                );
            }
        }
    }

    #[test]
    fn test_view_mapreduce_fusion() {
        // Test that View operations are fused into MapReduce operations
        // expand(unsqueeze(x)) + y should not generate separate View kernels

        let x = input(vec![Expr::Const(4), Expr::Const(8)], DType::F32).with_name("x");
        let y = input(
            vec![Expr::Const(4), Expr::Const(8), Expr::Const(16)],
            DType::F32,
        )
        .with_name("y");

        // Create expanded tensor through unsqueeze + expand
        // [4, 8] → [4, 8, 1] → [4, 8, 16]
        let expanded = x.unsqueeze(2).expand(2, Expr::Const(16));

        // Elementwise operation: expanded + y (each view is used once)
        let result = &expanded + &y;

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[result]).expect("Lowering should succeed");

        // Verify: should have only 1 kernel (the elementwise operation)
        // The View operations (unsqueeze, expand) should be inlined
        if let AstNode::Program { functions, .. } = &program {
            assert_eq!(
                functions.len(),
                1,
                "View operations should be fused into MapReduce, expected 1 kernel but got {}",
                functions.len()
            );

            // Verify the kernel is an elementwise operation (E_)
            if let AstNode::Kernel { name, .. } = &functions[0] {
                let kernel_name = name.as_ref().unwrap();
                assert!(
                    kernel_name.starts_with("E_"),
                    "Expected elementwise kernel, got: {}",
                    kernel_name
                );
            }
        }
    }

    #[test]
    fn test_view_mapreduce_fusion_with_reduction() {
        // Test View→MapReduce fusion with reduction: expand → mul → sum
        // This simulates a common matmul pattern

        let a = input(vec![Expr::Const(4), Expr::Const(8)], DType::F32).with_name("a");
        let b = input(vec![Expr::Const(8), Expr::Const(16)], DType::F32).with_name("b");

        // Expand a: [4, 8] → [4, 8, 1] → [4, 8, 16]
        let a_exp = a.unsqueeze(2).expand(2, Expr::Const(16));

        // Expand b: [8, 16] → [1, 8, 16] → [4, 8, 16]
        let b_exp = b.unsqueeze(0).expand(0, Expr::Const(4));

        // Multiply and reduce (matmul-like operation)
        let prod = &a_exp * &b_exp;
        let result = prod.sum(1); // [4, 16]

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[result]).expect("Lowering should succeed");

        // Should have 2 kernels: elementwise mul and reduce
        // The 4 View operations should all be inlined
        if let AstNode::Program { functions, .. } = &program {
            assert!(
                functions.len() <= 2,
                "View operations should be fused, expected at most 2 kernels but got {}",
                functions.len()
            );
        }
    }

    #[test]
    fn test_view_not_inlined_with_multiple_consumers() {
        // Test that Views with multiple consumers are NOT inlined
        // They require a buffer because the result is used multiple times

        let x = input(vec![Expr::Const(4), Expr::Const(8)], DType::F32).with_name("x");

        // Create expanded tensor
        let expanded = x.unsqueeze(2);

        // Use expanded tensor in two different operations
        let a = &expanded + &expanded; // consumer 1
        let b = &expanded * &expanded; // consumer 2

        // Combine results
        let result = &a + &b;

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[result]).expect("Lowering should succeed");

        // With multiple consumers, View should get its own kernel
        if let AstNode::Program { functions, .. } = &program {
            // Should have at least 2 kernels (View + combined elementwise ops)
            // The exact number depends on fusion, but View cannot be fully inlined
            assert!(
                functions.len() >= 2,
                "View with multiple consumers should not be fully inlined"
            );
        }
    }
}
