//! Main lowering logic for converting GraphNode to AST
//!
//! This module implements the core lowering algorithm that converts
//! computation graph nodes into executable AST kernels.

use std::collections::HashMap;

use crate::ast::{AstNode, DType, Literal, Mutability, VarDecl, VarKind};
use crate::graph::{Expr, GraphNode, GraphOp, ReduceOp, View, collect_inputs, topological_sort};

use super::fusion::AllFusions;
use super::fusion::FusionPass;
use super::index_gen::IndexGenerator;
use super::loop_gen::LoopGenerator;

/// Buffer information for kernel parameters
#[derive(Clone, Debug)]
struct BufferInfo {
    name: String,
    dtype: DType,
    shape: Vec<crate::graph::Expr>,
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
}

impl KernelOpType {
    /// Get the prefix character for kernel naming
    fn prefix(&self) -> &'static str {
        match self {
            KernelOpType::Elementwise => "E",
            KernelOpType::Reduce => "R",
            KernelOpType::Copy => "C",
            KernelOpType::Scatter => "S",
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
        }
    }

    /// Get the index generator
    pub fn index_gen(&self) -> &IndexGenerator {
        self.loop_gen.index_gen()
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
    pub fn lower(&mut self, roots: &[GraphNode]) -> AstNode {
        // Apply fusion passes
        let fused = AllFusions.apply(roots);

        // Get topological order
        let sorted = topological_sort(&fused);

        // Identify external inputs
        let inputs = collect_inputs(&fused);
        for input in &inputs {
            self.get_buffer_name(input);
        }

        // Generate kernels for each non-input node
        for node in sorted {
            if !node.is_external() {
                let kernel = self.lower_node(&node);
                self.kernels.push(kernel);
            }
        }

        // Create Program node
        AstNode::Program {
            functions: self.kernels.clone(),
            execution_waves: vec![], // TODO: dependency analysis
        }
    }

    /// Lower a single graph node to a Kernel AST
    fn lower_node(&mut self, node: &GraphNode) -> AstNode {
        let output_buf = self.get_buffer_name(node);

        match node.op() {
            GraphOp::View(_view) => {
                // View operation: just copy with index transformation
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
        }
    }

    /// Collect buffer info for input sources (deduplicated)
    fn collect_input_buffers(&self, node: &GraphNode) -> Vec<BufferInfo> {
        let mut seen = std::collections::HashSet::new();
        node.sources()
            .iter()
            .filter_map(|src| {
                let name = {
                    let ptr = std::rc::Rc::as_ptr(&src.0);
                    self.buffer_map.get(&ptr).cloned().unwrap_or_default()
                };
                // Only include if we haven't seen this buffer name before
                if seen.insert(name.clone()) {
                    Some(BufferInfo {
                        name,
                        dtype: self.get_buffer_dtype(src),
                        shape: src.shape().clone(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Lower a View operation
    fn lower_view(&mut self, node: &GraphNode, output_buf: &str) -> AstNode {
        let shape = node.shape();
        let output_idx = self.index_gen().view_to_index(node.view());

        // Generate tinygrad-style kernel name
        let kernel_name = self.generate_kernel_name(KernelOpType::Copy, node);

        // Get source buffer and index
        let src = &node.sources()[0];
        let src_buf = self.get_buffer_name(src);
        let src_idx = self.index_gen().view_to_index(src.view());

        // Collect buffer info
        let input_buffers = self.collect_input_buffers(node);
        let output_buffer = BufferInfo {
            name: output_buf.to_string(),
            dtype: node.dtype().clone(),
            shape: node.shape().clone(),
        };

        // Load from source, store to output
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Var(src_buf)),
            offset: Box::new(src_idx),
            count: 1,
            dtype: node.dtype().clone(),
        };

        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(output_idx),
            value: Box::new(load),
        };

        let body = self.loop_gen.generate_loops(&shape, store);

        self.make_kernel(&kernel_name, body, input_buffers, output_buffer)
    }

    /// Lower an elementwise (MapReduce with reduce=None) operation
    fn lower_elementwise(&mut self, node: &GraphNode, output_buf: &str, map: &AstNode) -> AstNode {
        let shape = node.shape();
        let output_idx = self.index_gen().view_to_index(node.view());

        // Generate tinygrad-style kernel name
        let kernel_name = self.generate_kernel_name(KernelOpType::Elementwise, node);

        // Collect buffer info
        let input_buffers = self.collect_input_buffers(node);
        let output_buffer = BufferInfo {
            name: output_buf.to_string(),
            dtype: node.dtype().clone(),
            shape: node.shape().clone(),
        };

        // Substitute Wildcards with Load operations
        let element_expr = self.substitute_wildcards(node, map);

        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(output_idx),
            value: Box::new(element_expr),
        };

        let body = self.loop_gen.generate_loops(&shape, store);

        self.make_kernel(&kernel_name, body, input_buffers, output_buffer)
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
        // Use source shape for iteration (before reduction)
        let src = &node.sources()[0];
        let src_shape = src.shape();

        // Generate tinygrad-style kernel name
        let kernel_name = self.generate_kernel_name(KernelOpType::Reduce, node);

        // Collect buffer info
        let input_buffers = self.collect_input_buffers(node);
        let output_buffer = BufferInfo {
            name: output_buf.to_string(),
            dtype: node.dtype().clone(),
            shape: node.shape().clone(),
        };

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

        self.make_kernel(&kernel_name, body, input_buffers, output_buffer)
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

        let src = &node.sources()[0];
        let src_shape = src.shape();
        let dtype = node.dtype().clone();

        // Generate kernel name
        let kernel_name = self.generate_kernel_name(KernelOpType::Scatter, node);

        // Collect buffer info
        let input_buffers = self.collect_input_buffers(node);
        let output_buffer = BufferInfo {
            name: output_buf.to_string(),
            dtype: dtype.clone(),
            shape: output_shape.to_vec(),
        };

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

        self.make_kernel(&kernel_name, body, input_buffers, output_buffer)
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

        // Create parameters for input buffers
        let mut params: Vec<VarDecl> = input_buffers
            .into_iter()
            .map(|buf| VarDecl {
                name: buf.name,
                dtype: DType::Ptr(Box::new(buf.dtype)),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            })
            .collect();

        // Add output buffer parameter
        params.push(VarDecl {
            name: output_buffer.name,
            dtype: DType::Ptr(Box::new(output_buffer.dtype)),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

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
    fn substitute_wildcards(&mut self, node: &GraphNode, expr: &AstNode) -> AstNode {
        match expr {
            AstNode::Wildcard(id) => {
                // Parse the wildcard ID to get source index
                let src_idx: usize = id.parse().unwrap_or(0);
                if src_idx < node.sources().len() {
                    let src = &node.sources()[src_idx];
                    let src_buf = self.get_buffer_name(src);
                    let idx = self.index_gen().view_to_index(src.view());

                    AstNode::Load {
                        ptr: Box::new(AstNode::Var(src_buf)),
                        offset: Box::new(idx),
                        count: 1,
                        dtype: src.dtype().clone(),
                    }
                } else {
                    expr.clone()
                }
            }
            // Recursively substitute in compound expressions
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Max(a, b) => AstNode::Max(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Recip(a) => AstNode::Recip(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Sqrt(a) => AstNode::Sqrt(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Log2(a) => AstNode::Log2(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Exp2(a) => AstNode::Exp2(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Sin(a) => AstNode::Sin(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Floor(a) => AstNode::Floor(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Cast(a, dtype) => {
                AstNode::Cast(Box::new(self.substitute_wildcards(node, a)), dtype.clone())
            }
            AstNode::Lt(a, b) => AstNode::Lt(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::And(a, b) => AstNode::And(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Not(a) => AstNode::Not(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Select {
                cond,
                then_val,
                else_val,
            } => AstNode::Select {
                cond: Box::new(self.substitute_wildcards(node, cond)),
                then_val: Box::new(self.substitute_wildcards(node, then_val)),
                else_val: Box::new(self.substitute_wildcards(node, else_val)),
            },
            // For other node types, return as-is
            _ => expr.clone(),
        }
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
        let program = lowerer.lower(&[y]);

        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_lower_reduction() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("x");
        let y = x.sum(1).with_name("y");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[y]);

        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_lower_complex_graph() {
        let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
        let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");

        // (a + b).sum(1)
        let c = (&a + &b).sum(1).with_name("c");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[c]);

        assert!(matches!(program, AstNode::Program { functions, .. } if !functions.is_empty()));
    }
}
