//! Graph node definitions for computation graph representation

use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::ast::{AstNode, DType, Literal};

use super::shape::{Expr, View};

// ============================================================================
// ReduceOp
// ============================================================================

/// Reduction operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Sum: Σ x
    Sum,
    /// Maximum: max(x)
    Max,
    /// Minimum: min(x)
    Min,
    /// Product: Π x
    Prod,
}

impl ReduceOp {
    /// Get the identity element for this reduction operation
    pub fn identity(&self, dtype: &DType) -> Literal {
        match self {
            ReduceOp::Sum => match dtype {
                DType::F32 => Literal::F32(0.0),
                DType::F64 => Literal::F64(0.0),
                DType::I32 => Literal::I32(0),
                DType::I64 => Literal::I64(0),
                _ => Literal::F32(0.0),
            },
            ReduceOp::Max => match dtype {
                DType::F32 => Literal::F32(f32::NEG_INFINITY),
                DType::F64 => Literal::F64(f64::NEG_INFINITY),
                DType::I32 => Literal::I32(i32::MIN),
                DType::I64 => Literal::I64(i64::MIN),
                _ => Literal::F32(f32::NEG_INFINITY),
            },
            ReduceOp::Min => match dtype {
                DType::F32 => Literal::F32(f32::INFINITY),
                DType::F64 => Literal::F64(f64::INFINITY),
                DType::I32 => Literal::I32(i32::MAX),
                DType::I64 => Literal::I64(i64::MAX),
                _ => Literal::F32(f32::INFINITY),
            },
            ReduceOp::Prod => match dtype {
                DType::F32 => Literal::F32(1.0),
                DType::F64 => Literal::F64(1.0),
                DType::I32 => Literal::I32(1),
                DType::I64 => Literal::I64(1),
                _ => Literal::F32(1.0),
            },
        }
    }

    /// Generate AstNode for combining accumulator with new value
    pub fn combine(&self, acc: AstNode, val: AstNode) -> AstNode {
        match self {
            ReduceOp::Sum => AstNode::Add(Box::new(acc), Box::new(val)),
            ReduceOp::Max => AstNode::Max(Box::new(acc), Box::new(val)),
            ReduceOp::Min => {
                // min(a, b) = select(a < b, a, b)
                AstNode::Select {
                    cond: Box::new(AstNode::Lt(Box::new(acc.clone()), Box::new(val.clone()))),
                    then_val: Box::new(acc),
                    else_val: Box::new(val),
                }
            }
            ReduceOp::Prod => AstNode::Mul(Box::new(acc), Box::new(val)),
        }
    }
}

// ============================================================================
// GraphOp
// ============================================================================

/// Graph operation type
#[derive(Debug, Clone)]
pub enum GraphOp {
    /// View transformation only (no data modification)
    ///
    /// The inner View represents the source view that this node applies to.
    View(View),

    /// Map-Reduce operation
    ///
    /// - `map`: Element-wise operation expressed as AstNode.
    ///   Input elements are referenced using `AstNode::Wildcard("0")`, `Wildcard("1")`, etc.
    ///   These are replaced with Load operations during lowering.
    /// - `reduce`: Optional reduction (operation, axis)
    MapReduce {
        /// Element-wise operation
        map: AstNode,
        /// Optional reduction: (operation, axis)
        reduce: Option<(ReduceOp, usize)>,
    },
}

// ============================================================================
// GraphInner
// ============================================================================

/// Internal data for a graph node
#[derive(Debug)]
pub struct GraphInner {
    /// Input source nodes (multiple allowed)
    /// Order matters: index corresponds to Wildcard("0"), Wildcard("1"), etc.
    pub src: Vec<GraphNode>,

    /// Output shape and memory layout
    pub view: View,

    /// Operation performed by this node
    pub op: GraphOp,

    /// Output data type
    pub dtype: DType,

    /// Optional name for debugging
    pub name: Option<String>,

    /// External buffer reference ID (for input/constant nodes)
    /// None: computed node (result from src)
    /// Some(id): references external buffer
    pub buffer_id: Option<usize>,
}

// ============================================================================
// GraphNode
// ============================================================================

/// Computation graph node (reference-counted)
///
/// Uses `Rc` to share `GraphInner`, enabling efficient DAG representation.
/// The same node can be referenced by multiple downstream nodes.
#[derive(Clone)]
pub struct GraphNode(pub Rc<GraphInner>);

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphNode")
            .field("name", &self.0.name)
            .field("dtype", &self.0.dtype)
            .field("shape", &self.shape())
            .finish()
    }
}

impl PartialEq for GraphNode {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for GraphNode {}

impl Hash for GraphNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl GraphNode {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a new graph node
    pub fn new(
        src: Vec<GraphNode>,
        view: View,
        op: GraphOp,
        dtype: DType,
        buffer_id: Option<usize>,
    ) -> Self {
        GraphNode(Rc::new(GraphInner {
            src,
            view,
            op,
            dtype,
            name: None,
            buffer_id,
        }))
    }

    // ========================================================================
    // Basic Accessors
    // ========================================================================

    /// Get the output shape
    pub fn shape(&self) -> Vec<Expr> {
        self.0.view.shape()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.view.ndim()
    }

    /// Get the data type
    pub fn dtype(&self) -> &DType {
        &self.0.dtype
    }

    /// Get the view
    pub fn view(&self) -> &View {
        &self.0.view
    }

    /// Get the operation
    pub fn op(&self) -> &GraphOp {
        &self.0.op
    }

    /// Get the source nodes
    pub fn sources(&self) -> &[GraphNode] {
        &self.0.src
    }

    /// Get the optional name
    pub fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    /// Check if this node references an external buffer
    pub fn is_external(&self) -> bool {
        self.0.buffer_id.is_some()
    }

    /// Get the buffer ID (for input/constant nodes)
    pub fn buffer_id(&self) -> Option<usize> {
        self.0.buffer_id
    }

    // ========================================================================
    // View Transformations
    // ========================================================================

    /// Permute dimensions
    pub fn permute(&self, axes: &[usize]) -> Self {
        let new_view = self.0.view.clone().permute(axes.to_vec());
        GraphNode::new(
            vec![self.clone()],
            new_view,
            GraphOp::View(self.0.view.clone()),
            self.0.dtype.clone(),
            None,
        )
    }

    /// Add a dimension of size 1
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let new_view = self.0.view.clone().unsqueeze(axis);
        GraphNode::new(
            vec![self.clone()],
            new_view,
            GraphOp::View(self.0.view.clone()),
            self.0.dtype.clone(),
            None,
        )
    }

    /// Remove a dimension of size 1
    pub fn squeeze(&self, axis: usize) -> Self {
        let new_view = self.0.view.clone().squeeze(axis);
        GraphNode::new(
            vec![self.clone()],
            new_view,
            GraphOp::View(self.0.view.clone()),
            self.0.dtype.clone(),
            None,
        )
    }

    /// Reshape to a new shape
    pub fn reshape(&self, shape: Vec<Expr>) -> Self {
        let new_view = self.0.view.clone().reshape(shape);
        GraphNode::new(
            vec![self.clone()],
            new_view,
            GraphOp::View(self.0.view.clone()),
            self.0.dtype.clone(),
            None,
        )
    }

    /// Flip a dimension
    pub fn flip(&self, axis: usize) -> Self {
        let new_view = self.0.view.clone().flip(axis);
        GraphNode::new(
            vec![self.clone()],
            new_view,
            GraphOp::View(self.0.view.clone()),
            self.0.dtype.clone(),
            None,
        )
    }

    /// Repeat/broadcast a dimension
    pub fn repeat(&self, axis: usize, times: Expr) -> Self {
        let new_view = self.0.view.clone().repeat(axis, times);
        GraphNode::new(
            vec![self.clone()],
            new_view,
            GraphOp::View(self.0.view.clone()),
            self.0.dtype.clone(),
            None,
        )
    }

    // ========================================================================
    // Reductions
    // ========================================================================

    /// Sum reduction along an axis
    pub fn sum(&self, axis: usize) -> Self {
        self.reduce(ReduceOp::Sum, axis)
    }

    /// Max reduction along an axis
    pub fn max(&self, axis: usize) -> Self {
        self.reduce(ReduceOp::Max, axis)
    }

    /// Min reduction along an axis
    pub fn min(&self, axis: usize) -> Self {
        self.reduce(ReduceOp::Min, axis)
    }

    /// Product reduction along an axis
    pub fn prod(&self, axis: usize) -> Self {
        self.reduce(ReduceOp::Prod, axis)
    }

    /// Generic reduction along an axis
    fn reduce(&self, op: ReduceOp, axis: usize) -> Self {
        // Calculate output shape (reduce dimension to 1)
        let mut new_shape = self.shape();
        new_shape[axis] = Expr::Const(1);
        let new_view = View::contiguous(new_shape);

        // Identity map: Wildcard("0") (just pass through the input)
        let map = AstNode::Wildcard("0".to_string());

        GraphNode::new(
            vec![self.clone()],
            new_view,
            GraphOp::MapReduce {
                map,
                reduce: Some((op, axis)),
            },
            self.0.dtype.clone(),
            None,
        )
    }

    // ========================================================================
    // Cast
    // ========================================================================

    /// Cast to a different data type
    pub fn cast(&self, dtype: DType) -> Self {
        let map = AstNode::Cast(Box::new(AstNode::Wildcard("0".to_string())), dtype.clone());

        GraphNode::new(
            vec![self.clone()],
            self.0.view.clone(),
            GraphOp::MapReduce { map, reduce: None },
            dtype,
            None,
        )
    }

    // ========================================================================
    // Naming
    // ========================================================================

    /// Attach a name for debugging
    pub fn with_name(self, name: impl Into<String>) -> Self {
        let mut inner = (*self.0).clone();
        inner.name = Some(name.into());
        GraphNode(Rc::new(inner))
    }

    /// Create a copy with new source nodes
    pub fn with_new_sources(&self, new_sources: Vec<GraphNode>) -> Self {
        let mut inner = (*self.0).clone();
        inner.src = new_sources;
        GraphNode(Rc::new(inner))
    }
}

// We need Clone for GraphInner to support with_name and with_new_sources
impl Clone for GraphInner {
    fn clone(&self) -> Self {
        GraphInner {
            src: self.src.clone(),
            view: self.view.clone(),
            op: self.op.clone(),
            dtype: self.dtype.clone(),
            name: self.name.clone(),
            buffer_id: self.buffer_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::shape::View;

    #[test]
    fn test_reduce_op_identity() {
        let sum_id = ReduceOp::Sum.identity(&DType::F32);
        assert!(matches!(sum_id, Literal::F32(v) if v == 0.0));

        let max_id = ReduceOp::Max.identity(&DType::F32);
        assert!(matches!(max_id, Literal::F32(v) if v == f32::NEG_INFINITY));
    }

    #[test]
    fn test_graph_node_shape() {
        let view = View::contiguous(vec![Expr::Const(32), Expr::Const(64)]);
        let empty_shape: Vec<Expr> = vec![];
        let node = GraphNode::new(
            vec![],
            view,
            GraphOp::View(View::contiguous(empty_shape)),
            DType::F32,
            Some(0),
        );

        assert_eq!(node.ndim(), 2);
        assert_eq!(node.shape().len(), 2);
    }

    #[test]
    fn test_graph_node_equality() {
        let view = View::contiguous(vec![Expr::Const(32)]);
        let empty_shape: Vec<Expr> = vec![];
        let node1 = GraphNode::new(
            vec![],
            view.clone(),
            GraphOp::View(View::contiguous(empty_shape.clone())),
            DType::F32,
            Some(0),
        );
        let node2 = node1.clone();
        let node3 = GraphNode::new(
            vec![],
            view,
            GraphOp::View(View::contiguous(empty_shape)),
            DType::F32,
            Some(1),
        );

        assert_eq!(node1, node2);
        assert_ne!(node1, node3);
    }
}
