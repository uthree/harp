//! Execution engine for computation graphs.
//!
//! This module provides the bridge between the Tensor API and the backend
//! execution pipeline. It handles graph lowering, kernel compilation, buffer
//! management, and execution.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::backend::traits::Buffer;
use crate::graph::GraphNode;

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
/// * `inputs` - Pre-allocated input buffers, keyed by buffer ID
///
/// # Returns
///
/// An `ExecutionResult` containing the output buffers.
///
/// # Errors
///
/// Returns an error if:
/// - No device is set
/// - Compilation fails
/// - Execution fails
/// - Required input buffers are missing
pub fn execute_graph(
    _roots: &[GraphNode],
    _inputs: &HashMap<usize, &dyn Buffer>,
) -> Result<ExecutionResult, ExecutionError> {
    use crate::backend::DeviceKind;
    use crate::backend::global::get_default_device_kind;

    // 1. Check device is set
    let device_kind = get_default_device_kind();
    if device_kind == DeviceKind::None {
        return Err(ExecutionError::NoDevice);
    }

    // TODO: Implement the full execution pipeline
    // For now, return an error indicating not yet implemented
    Err(ExecutionError::Internal(
        "execute_graph is not yet fully implemented".to_string(),
    ))
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
