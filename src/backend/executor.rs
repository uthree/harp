//! Execution engine for computation graphs.
//!
//! This module provides the bridge between the Tensor API and the backend
//! execution pipeline. It handles graph lowering, kernel compilation, buffer
//! management, and execution.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

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
/// 3. Compiles the AST to kernel(s)
/// 4. Allocates output buffers
/// 5. Executes the kernel(s)
/// 6. Returns the output buffers
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
    use crate::backend::DeviceKind;
    use crate::backend::global::{
        allocate_buffer_on_default_device, compile_ast_on_default_device, get_default_device_kind,
    };
    use crate::lowerer::Lowerer;

    // 1. Check device is set
    let device_kind = get_default_device_kind();
    if device_kind == DeviceKind::None {
        return Err(ExecutionError::NoDevice);
    }

    if roots.is_empty() {
        return Ok(ExecutionResult::new());
    }

    // 2. Lower graph to AST
    let mut lowerer = Lowerer::new();
    let program = lowerer.lower(roots);

    // 3. Collect input nodes and build signature
    let input_nodes = collect_inputs(roots);
    let signature = lowerer.build_kernel_signature(&input_nodes, roots);

    // 4. Verify all inputs have buffers
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

    // 5. Compile AST to kernel
    let kernel = compile_ast_on_default_device(program, signature).map_err(|e| {
        ExecutionError::CompilationFailed(format!("Failed to compile kernel: {}", e))
    })?;

    // 6. Prepare input buffer references in order
    let mut ordered_inputs: Vec<&dyn Buffer> = Vec::with_capacity(input_nodes.len());
    for input_node in &input_nodes {
        let ptr = std::rc::Rc::as_ptr(&input_node.0);
        let buffer = input_buffers.get(&ptr).unwrap();
        ordered_inputs.push(buffer.as_ref());
    }

    // 7. Allocate output buffers
    let mut output_buffers: Vec<Box<dyn Buffer>> = Vec::with_capacity(roots.len());
    for root in roots {
        let shape: Vec<usize> = root
            .shape()
            .iter()
            .map(|e| {
                e.evaluate()
                    .expect("Cannot evaluate shape expression at runtime") as usize
            })
            .collect();
        let dtype = root.dtype().clone();

        let buffer = allocate_buffer_on_default_device(shape, dtype).map_err(|e| {
            ExecutionError::AllocationFailed(format!("Failed to allocate output buffer: {}", e))
        })?;
        output_buffers.push(buffer);
    }

    // 8. Execute kernel with output buffers
    // We need to create a Vec<&mut dyn Buffer> manually to avoid lifetime issues
    let output_refs: &mut [&mut dyn Buffer] = &mut output_buffers
        .iter_mut()
        .map(|b| &mut **b as &mut dyn Buffer)
        .collect::<Vec<_>>();

    kernel
        .execute(&ordered_inputs, output_refs)
        .map_err(|e| ExecutionError::ExecutionFailed(format!("Kernel execution failed: {}", e)))?;

    // 9. Build result
    let mut result = ExecutionResult::new();
    for (i, buffer) in output_buffers.into_iter().enumerate() {
        result.insert(i, buffer);
    }

    Ok(result)
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
