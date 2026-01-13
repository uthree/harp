//! Graph node construction utilities

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ast::DType;

use super::node::{GraphNode, GraphOp};
use super::shape::{Expr, View};

/// Buffer ID counter for generating unique IDs
static BUFFER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generate a new unique buffer ID
fn next_buffer_id() -> usize {
    BUFFER_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Reset buffer ID counter (for testing)
#[cfg(test)]
#[allow(dead_code)]
pub fn reset_buffer_id_counter() {
    BUFFER_ID_COUNTER.store(0, Ordering::SeqCst);
}

// ============================================================================
// Input Constructors
// ============================================================================

/// Create an input tensor (placeholder)
///
/// # Arguments
/// * `shape` - Tensor shape as a vector of Expr
/// * `dtype` - Data type
///
/// # Example
/// ```
/// use eclat::graph::{input, Expr, DType};
/// let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
/// ```
pub fn input(shape: Vec<Expr>, dtype: DType) -> GraphNode {
    let view = View::contiguous(shape);
    GraphNode::new(
        vec![],
        view.clone(),
        GraphOp::View(view),
        dtype,
        Some(next_buffer_id()),
    )
}

/// Create a named input tensor
pub fn named_input(name: impl Into<String>, shape: Vec<Expr>, dtype: DType) -> GraphNode {
    input(shape, dtype).with_name(name)
}

/// Create an input tensor with dynamic shape
///
/// Uses Expr::Idx as placeholders for dynamic dimensions.
/// The actual dimension values will be resolved at runtime.
///
/// # Example
/// ```
/// use eclat::graph::{dynamic_input, DType};
/// // Tensor with 4 dynamic dimensions (indexed 0..3)
/// let x = dynamic_input(4, DType::F32);
/// ```
pub fn dynamic_input(ndim: usize, dtype: DType) -> GraphNode {
    // Use Expr::Idx as placeholders for dynamic dimensions
    // These will be substituted with actual values during lowering
    let shape = (0..ndim).map(Expr::Idx).collect();
    input(shape, dtype)
}

// ============================================================================
// Constant Constructors
// ============================================================================

/// Create a constant tensor (external buffer reference)
///
/// The actual data is referenced by buffer ID and resolved at runtime.
pub fn constant(shape: Vec<Expr>, dtype: DType) -> GraphNode {
    let view = View::contiguous(shape);
    GraphNode::new(
        vec![],
        view.clone(),
        GraphOp::View(view),
        dtype,
        Some(next_buffer_id()),
    )
    .with_name("const")
}

/// Create a zero-initialized tensor
pub fn zeros(shape: Vec<Expr>, dtype: DType) -> GraphNode {
    constant(shape, dtype).with_name("zeros")
}

/// Create a tensor initialized with ones
pub fn ones(shape: Vec<Expr>, dtype: DType) -> GraphNode {
    constant(shape, dtype).with_name("ones")
}

/// Create a scalar constant
pub fn scalar(dtype: DType) -> GraphNode {
    let empty_shape: Vec<Expr> = vec![]; // 0-dimensional
    let view = View::contiguous(empty_shape);
    GraphNode::new(
        vec![],
        view.clone(),
        GraphOp::View(view),
        dtype,
        Some(next_buffer_id()),
    )
    .with_name("scalar")
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Builder for creating GraphNodes with fluent API
pub struct GraphNodeBuilder {
    shape: Vec<Expr>,
    dtype: DType,
    name: Option<String>,
}

impl GraphNodeBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            shape: vec![],
            dtype: DType::F32,
            name: None,
        }
    }

    /// Set the shape
    pub fn shape(mut self, shape: Vec<Expr>) -> Self {
        self.shape = shape;
        self
    }

    /// Set the shape from constant dimensions
    pub fn shape_const(self, dims: &[i64]) -> Self {
        let shape: Vec<Expr> = dims.iter().map(|&d| Expr::Const(d)).collect();
        Self { shape, ..self }
    }

    /// Set the data type
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Build as an input tensor
    pub fn build_input(self) -> GraphNode {
        let node = input(self.shape, self.dtype);
        if let Some(name) = self.name {
            node.with_name(name)
        } else {
            node
        }
    }

    /// Build as a constant tensor
    pub fn build_constant(self) -> GraphNode {
        let node = constant(self.shape, self.dtype);
        if let Some(name) = self.name {
            node.with_name(name)
        } else {
            node
        }
    }
}

impl Default for GraphNodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience Macros
// ============================================================================

/// Macro for creating a shape from integer literals
///
/// # Example
/// ```
/// use eclat::graph_shape;
/// let shape = graph_shape![32, 64, 128];
/// ```
#[macro_export]
macro_rules! graph_shape {
    ($($dim:expr),* $(,)?) => {
        vec![$($crate::graph::Expr::Const($dim as i64)),*]
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_creation() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        assert_eq!(x.ndim(), 2);
        assert!(x.is_external());
        // buffer_id is auto-incremented, just check it exists
        assert!(x.buffer_id().is_some());
    }

    #[test]
    fn test_named_input() {
        let x = named_input("my_input", vec![Expr::Const(32)], DType::F32);
        assert_eq!(x.name(), Some("my_input"));
    }

    #[test]
    fn test_dynamic_input() {
        let x = dynamic_input(2, DType::F32);
        assert_eq!(x.ndim(), 2);
    }

    #[test]
    fn test_constant() {
        let c = constant(vec![Expr::Const(10)], DType::I32);
        assert!(c.is_external());
        assert_eq!(c.name(), Some("const"));
    }

    #[test]
    fn test_builder() {
        let x = GraphNodeBuilder::new()
            .shape_const(&[32, 64])
            .dtype(DType::F64)
            .name("my_tensor")
            .build_input();

        assert_eq!(x.ndim(), 2);
        assert_eq!(x.dtype(), &DType::F64);
        assert_eq!(x.name(), Some("my_tensor"));
    }

    #[test]
    fn test_graph_shape_macro() {
        let shape = graph_shape![32, 64, 128];
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], Expr::Const(32));
    }
}
