//! Execution engine for computation graphs.
//!
//! This module provides the bridge between the Tensor API and the backend
//! execution pipeline. It handles graph lowering, kernel compilation, buffer
//! management, and execution.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::backend::compile::CompilationPipeline;
use crate::backend::traits::Buffer;
use crate::graph::{GraphInner, GraphNode, collect_inputs};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during graph execution.
#[derive(Debug)]
pub enum ExecutionError {
    /// No device has been set. Call `set_device()` first.
    NoDevice,

    /// Device type mismatch between buffers or operations.
    DeviceMismatch(String),

    /// Kernel compilation failed.
    CompilationFailed(String),

    /// Kernel execution failed.
    ExecutionFailed(String),

    /// Buffer allocation failed.
    AllocationFailed(String),

    /// Required input buffer was not provided.
    MissingInput(String),

    /// Tensor has not been realized yet.
    NotRealized,

    /// Shape mismatch between expected and actual.
    ShapeMismatch(String),

    /// Data type mismatch.
    DTypeMismatch(String),

    /// Internal error.
    Internal(String),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => write!(f, "No device is set. Call set_device() first."),
            Self::DeviceMismatch(msg) => write!(f, "Device mismatch: {}", msg),
            Self::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            Self::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            Self::AllocationFailed(msg) => write!(f, "Buffer allocation failed: {}", msg),
            Self::MissingInput(name) => write!(f, "Missing input buffer: {}", name),
            Self::NotRealized => write!(f, "Tensor has not been realized"),
            Self::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            Self::DTypeMismatch(msg) => write!(f, "DType mismatch: {}", msg),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl Error for ExecutionError {}

impl From<Box<dyn Error + Send + Sync>> for ExecutionError {
    fn from(err: Box<dyn Error + Send + Sync>) -> Self {
        ExecutionError::ExecutionFailed(err.to_string())
    }
}

// ============================================================================
// Execution Result
// ============================================================================

/// Result of executing a computation graph.
///
/// Contains the output buffers keyed by their buffer IDs from the graph.
pub struct ExecutionResult {
    /// Output buffers mapped by buffer ID.
    pub buffers: HashMap<usize, Box<dyn Buffer>>,
}

impl ExecutionResult {
    /// Create a new execution result.
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Add a buffer to the result.
    pub fn insert(&mut self, buffer_id: usize, buffer: Box<dyn Buffer>) {
        self.buffers.insert(buffer_id, buffer);
    }

    /// Get a buffer by ID.
    pub fn get(&self, buffer_id: usize) -> Option<&dyn Buffer> {
        self.buffers.get(&buffer_id).map(|b| b.as_ref())
    }

    /// Take ownership of a buffer by ID.
    pub fn take(&mut self, buffer_id: usize) -> Option<Box<dyn Buffer>> {
        self.buffers.remove(&buffer_id)
    }
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Execute Graph Function
// ============================================================================

/// Execute a computation graph and return the output buffers.
///
/// This function:
/// 1. Validates that a device is set
/// 2. Lowers the graph to AST
/// 3. Compiles and executes each kernel in sequence
/// 4. Manages intermediate buffers
/// 5. Returns the output buffers
///
/// # Arguments
///
/// * `roots` - The root nodes of the computation graph to execute
/// * `input_buffers` - Pre-allocated input buffers, keyed by GraphNode pointer
///
/// # Returns
///
/// An `ExecutionResult` containing the output buffers, keyed by output index.
///
/// # Errors
///
/// Returns an error if:
/// - No device is set
/// - Compilation fails
/// - Execution fails
/// - Required input buffers are missing
pub fn execute_graph(
    roots: &[GraphNode],
    input_buffers: &HashMap<*const GraphInner, Box<dyn Buffer>>,
) -> Result<ExecutionResult, ExecutionError> {
    use crate::ast::AstNode;
    use crate::backend::DeviceKind;
    use crate::backend::global::{allocate_buffer_on_default_device, get_default_device_kind};

    // 1. Check device is set
    let device_kind = get_default_device_kind();
    if device_kind == DeviceKind::None {
        return Err(ExecutionError::NoDevice);
    }

    if roots.is_empty() {
        return Ok(ExecutionResult::new());
    }

    // 2. Lower and optimize graph using CompilationPipeline
    let pipeline = CompilationPipeline::from_default_device();
    let (ast, lowerer) = pipeline.lower_with_lowerer(roots);
    let program = pipeline.optimize(ast);

    // 3. Extract kernel info from the program
    let kernels = match &program {
        AstNode::Program { functions, .. } => functions.clone(),
        _ => {
            return Err(ExecutionError::Internal(
                "Expected Program node".to_string(),
            ));
        }
    };

    #[cfg(debug_assertions)]
    {
        eprintln!("[execute_graph] Generated {} kernels:", kernels.len());
        for (i, f) in kernels.iter().enumerate() {
            if let AstNode::Kernel { name, params, .. } = f {
                let param_names: Vec<_> = params.iter().map(|p| &p.name).collect();
                eprintln!("  Kernel {}: name={:?}, params={:?}", i, name, param_names);
            }
        }
    }

    // 4. Collect input nodes and verify buffers
    let input_nodes = collect_inputs(roots);

    #[cfg(debug_assertions)]
    {
        eprintln!("[execute_graph] Input nodes: {}", input_nodes.len());
        for (i, node) in input_nodes.iter().enumerate() {
            let buf_name = lowerer
                .lookup_buffer_name(node)
                .cloned()
                .unwrap_or_default();
            eprintln!(
                "  Input {}: name={:?}, buf_name={}, shape={:?}",
                i,
                node.name(),
                buf_name,
                node.shape()
            );
        }
    }

    for input_node in &input_nodes {
        let ptr = std::rc::Rc::as_ptr(&input_node.0);
        if !input_buffers.contains_key(&ptr) {
            let name = lowerer
                .lookup_buffer_name(input_node)
                .cloned()
                .unwrap_or_else(|| format!("node@{:p}", ptr));
            return Err(ExecutionError::MissingInput(name));
        }
    }

    // 5. Build a name-to-buffer map for input buffers
    let mut buffer_map: HashMap<String, Box<dyn Buffer>> = HashMap::new();
    for input_node in &input_nodes {
        let ptr = std::rc::Rc::as_ptr(&input_node.0);
        let name = lowerer
            .lookup_buffer_name(input_node)
            .cloned()
            .unwrap_or_default();
        let buffer = input_buffers.get(&ptr).unwrap().clone_buffer();
        buffer_map.insert(name, buffer);
    }

    // 6. Execute each kernel in sequence
    for kernel_ast in &kernels {
        if let AstNode::Kernel {
            name: kernel_name,
            params,
            default_grid_size,
            ..
        } = kernel_ast
        {
            let entry_point = kernel_name.clone().unwrap_or_else(|| "kernel".to_string());

            // Extract buffer parameters only (Ptr type)
            // Non-buffer parameters (LocalId, GroupId, shape vars) are handled separately
            let is_buffer_param =
                |p: &&crate::ast::VarDecl| matches!(p.dtype, crate::ast::DType::Ptr(_));

            let input_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.mutability, crate::ast::Mutability::Immutable))
                .filter(is_buffer_param)
                .collect();

            let output_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.mutability, crate::ast::Mutability::Mutable))
                .filter(is_buffer_param)
                .collect();

            // Build kernel inputs
            let mut kernel_inputs: Vec<&dyn Buffer> = Vec::new();
            for param in &input_params {
                let buf = buffer_map.get(&param.name).ok_or_else(|| {
                    ExecutionError::MissingInput(format!(
                        "Buffer '{}' not found for kernel '{}'",
                        param.name, entry_point
                    ))
                })?;
                kernel_inputs.push(buf.as_ref());
            }

            // Allocate output buffers
            let mut kernel_outputs: Vec<Box<dyn Buffer>> = Vec::new();
            for param in &output_params {
                // Try to get buffer info from lowerer first
                let (shape, dtype) =
                    if let Some((shape_exprs, dt)) = lowerer.get_buffer_info_by_name(&param.name) {
                        // Evaluate shape expressions to concrete sizes
                        let shape: Vec<usize> = shape_exprs
                            .iter()
                            .map(|e| {
                                e.evaluate()
                                    .expect("Cannot evaluate shape expression at runtime")
                                    as usize
                            })
                            .collect();
                        (shape, dt)
                    } else {
                        // Fallback to inferring from grid size and param dtype
                        let shape = infer_buffer_shape(default_grid_size);
                        let dtype = extract_buffer_dtype(&param.dtype);
                        (shape, dtype)
                    };

                #[cfg(debug_assertions)]
                eprintln!(
                    "[execute_graph] Allocating output buffer '{}' with shape {:?}, dtype {:?}",
                    param.name, shape, dtype
                );

                let buffer = allocate_buffer_on_default_device(shape, dtype).map_err(|e| {
                    ExecutionError::AllocationFailed(format!(
                        "Failed to allocate buffer '{}': {}",
                        param.name, e
                    ))
                })?;
                kernel_outputs.push(buffer);
            }

            // Compile this kernel
            let kernel_program = AstNode::Program {
                functions: vec![kernel_ast.clone()],
                execution_waves: vec![],
            };

            // Build signature for this kernel
            let kernel_sig = build_kernel_signature_from_params(&input_params, &output_params);

            let cache_entry =
                crate::backend::global::compile_ast_with_cache(kernel_program, kernel_sig)
                    .map_err(|e| {
                        ExecutionError::CompilationFailed(format!(
                            "Failed to compile kernel '{}': {}",
                            entry_point, e
                        ))
                    })?;
            let compiled_kernel = cache_entry.kernel;

            // Execute the kernel
            let output_refs: &mut [&mut dyn Buffer] = &mut kernel_outputs
                .iter_mut()
                .map(|b| &mut **b as &mut dyn Buffer)
                .collect::<Vec<_>>();

            compiled_kernel
                .execute(&kernel_inputs, output_refs)
                .map_err(|e| {
                    ExecutionError::ExecutionFailed(format!(
                        "Kernel '{}' execution failed: {}",
                        entry_point, e
                    ))
                })?;

            // Store output buffers in buffer_map for subsequent kernels
            for (param, buffer) in output_params.iter().zip(kernel_outputs.into_iter()) {
                #[cfg(debug_assertions)]
                {
                    if let Ok(data) = buffer.read_to_host() {
                        let len = std::cmp::min(data.len(), 32);
                        eprintln!(
                            "[execute_graph] After kernel '{}', buffer '{}': {} bytes, first bytes: {:?}",
                            entry_point,
                            param.name,
                            data.len(),
                            &data[..len]
                        );
                    }
                }
                buffer_map.insert(param.name.clone(), buffer);
            }
        }
    }

    // 7. Collect final output buffers
    // The root node's buffer should have been updated by the last kernel
    let mut result = ExecutionResult::new();

    #[cfg(debug_assertions)]
    {
        eprintln!("[execute_graph] Collecting output buffers:");
        eprintln!(
            "  Available buffers: {:?}",
            buffer_map.keys().collect::<Vec<_>>()
        );
    }

    for (i, root) in roots.iter().enumerate() {
        let buf_name = lowerer
            .lookup_buffer_name(root)
            .cloned()
            .unwrap_or_else(|| format!("output_{}", i));

        #[cfg(debug_assertions)]
        eprintln!("  Root {}: looking for buffer '{}'", i, buf_name);

        if let Some(buffer) = buffer_map.remove(&buf_name) {
            #[cfg(debug_assertions)]
            {
                if let Ok(data) = buffer.read_to_host() {
                    let len = std::cmp::min(data.len(), 32);
                    eprintln!(
                        "  -> Found buffer '{}': {} bytes, first bytes: {:?}",
                        buf_name,
                        data.len(),
                        &data[..len]
                    );
                }
            }
            result.insert(i, buffer);
        } else {
            // The output buffer might be the last one generated
            // Find the highest numbered buffer
            let highest_buf = buffer_map
                .keys()
                .filter(|k| k.starts_with("buf"))
                .filter_map(|k| {
                    k.strip_prefix("buf")
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|n| (n, k.clone()))
                })
                .max_by_key(|(n, _)| *n)
                .map(|(_, name)| name);

            #[cfg(debug_assertions)]
            eprintln!("  -> Not found, highest buffer: {:?}", highest_buf);

            if let Some(key) = highest_buf
                && let Some(buffer) = buffer_map.remove(&key)
            {
                #[cfg(debug_assertions)]
                {
                    if let Ok(data) = buffer.read_to_host() {
                        let len = std::cmp::min(data.len(), 32);
                        eprintln!(
                            "  -> Using buffer '{}': {} bytes, first bytes: {:?}",
                            key,
                            data.len(),
                            &data[..len]
                        );
                    }
                }
                result.insert(i, buffer);
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Infer buffer shape from kernel grid size
fn infer_buffer_shape(grid_size: &[Box<crate::ast::AstNode>; 3]) -> Vec<usize> {
    let mut shape = Vec::new();
    for dim in grid_size {
        if let crate::ast::AstNode::Const(crate::ast::Literal::I64(n)) = dim.as_ref()
            && *n > 1
        {
            shape.push(*n as usize);
        }
    }
    if shape.is_empty() {
        shape.push(1);
    }
    shape
}

/// Extract element dtype from pointer type
fn extract_buffer_dtype(dtype: &crate::ast::DType) -> crate::ast::DType {
    match dtype {
        crate::ast::DType::Ptr(inner) => inner.as_ref().clone(),
        other => other.clone(),
    }
}

/// Build a kernel signature from parameter declarations
fn build_kernel_signature_from_params(
    input_params: &[&crate::ast::VarDecl],
    output_params: &[&crate::ast::VarDecl],
) -> crate::backend::KernelSignature {
    use crate::backend::{BufferSignature, KernelSignature};

    let inputs: Vec<BufferSignature> = input_params
        .iter()
        .map(|p| BufferSignature::new(p.name.clone(), vec![]))
        .collect();

    let outputs: Vec<BufferSignature> = output_params
        .iter()
        .map(|p| BufferSignature::new(p.name.clone(), vec![]))
        .collect();

    KernelSignature::new(inputs, outputs)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_error_display() {
        let err = ExecutionError::NoDevice;
        assert!(err.to_string().contains("No device"));

        let err = ExecutionError::MissingInput("input_0".to_string());
        assert!(err.to_string().contains("input_0"));
    }

    #[test]
    fn test_execution_result() {
        let result = ExecutionResult::new();
        assert!(result.buffers.is_empty());
    }
}
