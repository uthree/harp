//! Initialization operations (primops)
//!
//! - Const: constant tensor
//! - Rand: uniform random [0, 1)

use std::marker::PhantomData;

use crate::ast::Literal;
use crate::graph::shape::{Expr, View};
use crate::graph::{DType, GraphNode, GraphOp};
use crate::tensor::{Dim, DimDyn, Dimension, Tensor, TensorNode, TensorOp};

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

// ============================================================================
// Static dimension constructors
// ============================================================================

impl<const N: usize> Tensor<Dim<N>>
where
    Dim<N>: Dimension,
{
    /// Create a tensor filled with zeros (Const(0))
    pub fn zeros(shape: [usize; N]) -> Self {
        Self::full(shape, 0.0)
    }

    /// Create a tensor filled with ones (Const(1))
    pub fn ones(shape: [usize; N]) -> Self {
        Self::full(shape, 1.0)
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: [usize; N], value: f32) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(
            TensorOp::ConstFill(Literal::F32(value)),
            vec![],
            view,
            DType::F32,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create an input tensor (placeholder for data)
    pub fn input(name: &str, shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
            DType::F32,
            name,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create a tensor with uniform random values [0, 1)
    pub fn rand(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(TensorOp::Rand, vec![], view, DType::F32);
        Self::from_tensor_node(tensor_node, shape_vec)
    }
}

// ============================================================================
// Dynamic dimension constructors
// ============================================================================

impl Tensor<DimDyn> {
    /// Create a tensor filled with zeros (dynamic shape)
    pub fn zeros_dyn(shape: &[usize]) -> Self {
        Self::full_dyn(shape, 0.0)
    }

    /// Create a tensor filled with ones (dynamic shape)
    pub fn ones_dyn(shape: &[usize]) -> Self {
        Self::full_dyn(shape, 1.0)
    }

    /// Create a tensor filled with a constant value (dynamic shape)
    pub fn full_dyn(shape: &[usize], value: f32) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(
            TensorOp::ConstFill(Literal::F32(value)),
            vec![],
            view,
            DType::F32,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create an input tensor (dynamic shape)
    pub fn input_dyn(name: &str, shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new_named(
            TensorOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
            DType::F32,
            name,
        );
        Self::from_tensor_node(tensor_node, shape_vec)
    }

    /// Create a tensor with uniform random values [0, 1) (dynamic shape)
    pub fn rand_dyn(shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let view = view_from_shape(&shape_vec);
        let tensor_node = TensorNode::new(TensorOp::Rand, vec![], view, DType::F32);
        Self::from_tensor_node(tensor_node, shape_vec)
    }
}

// ============================================================================
// Internal constructor
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Create a new tensor from a graph node (internal use)
    ///
    /// During transition period, this also creates a TensorNode from the GraphNode.
    /// The TensorNode.src is empty as we don't have access to source Tensors here.
    /// This will be fixed in Phase 3 when primops are updated to pass source Tensors.
    pub(crate) fn from_node(node: GraphNode, shape: Vec<usize>, dtype: DType) -> Self {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Create TensorNode from GraphNode (transition period)
        let tensor_op = TensorOp::from_graph_op(&node.op);
        let tensor_node = TensorNode::new(
            tensor_op,
            vec![], // Empty src during transition - will be fixed in Phase 3
            node.view.clone(),
            dtype.clone(),
        );

        Self {
            node,
            inner: Rc::new(tensor_node),
            shape,
            dtype,
            autograd: None,
            buffer: RefCell::new(None),
            _dim: PhantomData,
        }
    }

    /// Create a new tensor from a TensorNode directly (new API)
    ///
    /// This is the preferred method for Phase 3+ migration.
    /// The GraphNode is still created for backward compatibility but will be removed later.
    pub(crate) fn from_tensor_node(tensor_node: TensorNode, shape: Vec<usize>) -> Self {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Extract GraphNode sources from TensorNode sources
        let graph_sources: Vec<GraphNode> =
            tensor_node.src.iter().map(|t| t.node.clone()).collect();

        // Convert TensorOp to GraphOp for backward compatibility
        let graph_op = tensor_op_to_graph_op(&tensor_node.op, &tensor_node.view);

        let node = GraphNode::new(
            tensor_node.dtype.clone(),
            graph_op,
            graph_sources,
            tensor_node.view.clone(),
        );

        let dtype = tensor_node.dtype.clone();
        Self {
            node,
            inner: Rc::new(tensor_node),
            shape,
            dtype,
            autograd: None,
            buffer: RefCell::new(None),
            _dim: PhantomData,
        }
    }
}

/// Convert TensorOp to GraphOp for backward compatibility
///
/// This function is used during the transition period to create GraphNodes
/// from TensorNodes. It will be removed when graphモジュール is deleted.
fn tensor_op_to_graph_op(op: &TensorOp, view: &View) -> GraphOp {
    use crate::graph::ops::{ElementwiseOp as GE, ReduceOp as GR};

    match op {
        TensorOp::Buffer { name } => GraphOp::Buffer { name: name.clone() },
        TensorOp::Const(lit) => GraphOp::Const(lit.clone()),
        TensorOp::ConstFill(lit) => GraphOp::ConstFill(lit.clone()),
        TensorOp::Rand => GraphOp::Rand,
        TensorOp::Arange => GraphOp::Arange,
        TensorOp::Cast { target_dtype } => GraphOp::Cast {
            target_dtype: target_dtype.clone(),
        },
        TensorOp::Clone => GraphOp::Clone,
        TensorOp::View => GraphOp::View(view.clone()),
        TensorOp::Contiguous => GraphOp::Contiguous,
        TensorOp::Elementwise { op } => GraphOp::Elementwise {
            op: match op {
                crate::tensor::ElementwiseOp::Add => GE::Add,
                crate::tensor::ElementwiseOp::Mul => GE::Mul,
                crate::tensor::ElementwiseOp::Max => GE::Max,
                crate::tensor::ElementwiseOp::Rem => GE::Rem,
                crate::tensor::ElementwiseOp::Idiv => GE::Idiv,
                crate::tensor::ElementwiseOp::Neg => GE::Neg,
                crate::tensor::ElementwiseOp::Recip => GE::Recip,
                crate::tensor::ElementwiseOp::Log2 => GE::Log2,
                crate::tensor::ElementwiseOp::Exp2 => GE::Exp2,
                crate::tensor::ElementwiseOp::Sin => GE::Sin,
                crate::tensor::ElementwiseOp::Sqrt => GE::Sqrt,
                crate::tensor::ElementwiseOp::Floor => GE::Floor,
            },
        },
        TensorOp::FusedElementwise { expr } => GraphOp::FusedElementwise { expr: expr.clone() },
        TensorOp::Reduce { op, axes, .. } => GraphOp::Reduce {
            op: match op {
                crate::tensor::ReduceOp::Sum => GR::Sum,
                crate::tensor::ReduceOp::Prod => GR::Prod,
                crate::tensor::ReduceOp::Max => GR::Max,
            },
            axis: axes.first().copied().unwrap_or(0),
            reduce_strategy: None,
        },
        TensorOp::FusedElementwiseReduce {
            expr,
            reduce_op,
            axes,
            ..
        } => GraphOp::FusedElementwiseReduce {
            expr: expr.clone(),
            reduce_op: match reduce_op {
                crate::tensor::ReduceOp::Sum => GR::Sum,
                crate::tensor::ReduceOp::Prod => GR::Prod,
                crate::tensor::ReduceOp::Max => GR::Max,
            },
            axes: axes.clone(),
            reduce_strategy: None,
        },
        TensorOp::Pad { padding, value } => GraphOp::Pad {
            padding: padding.clone(),
            value: *value,
        },
        TensorOp::Slice { ranges } => GraphOp::Slice {
            ranges: ranges.clone(),
        },
        TensorOp::Concat { axis } => GraphOp::Concat { axis: *axis },
        TensorOp::Executed => GraphOp::Contiguous, // Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim1, Dim2, Dim3};

    #[test]
    fn test_zeros() {
        let t = Tensor::<Dim2>::zeros([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_ones() {
        let t = Tensor::<Dim3>::ones([2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_full() {
        let t = Tensor::<Dim1>::full([10], 2.5);
        assert_eq!(t.shape(), &[10]);
    }

    #[test]
    fn test_rand() {
        let t = Tensor::<Dim2>::rand([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_zeros_dyn() {
        let t = Tensor::<DimDyn>::zeros_dyn(&[3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
    }

    #[test]
    fn test_rand_dyn() {
        let t = Tensor::<DimDyn>::rand_dyn(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }
}
