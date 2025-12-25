//! Unified Tensor type for Harp
//!
//! This module provides a high-level `Tensor<D>` type that combines:
//! - Lazy evaluation via computation graphs
//! - Static dimension checking via const generics
//! - Automatic differentiation (autograd)
//!
//! # Design Philosophy
//!
//! The Tensor type follows the design philosophy of tinygrad/micrograd:
//! minimal primitives combined to create complex functionality.
//!
//! ## Primitive Operations
//!
//! **Binary**: Add, Mul, Max
//! **Unary**: Neg, Recip, Exp2, Log2, Sin, Sqrt
//! **Movement**: Reshape, Permute, Expand, Shrink, Pad
//! **Reduce**: ReduceSum, ReduceMax
//!
//! # Examples
//!
//! ```ignore
//! use harp::tensor::{Tensor, Dim2};
//!
//! // Create a 2D tensor with gradient tracking
//! let x = Tensor::<Dim2>::zeros([3, 4]).requires_grad();
//!
//! // Lazy operations (not executed yet)
//! let y = &x + &x;
//! let z = y.relu();
//!
//! // Execute computation
//! z.forward();
//!
//! // Get data
//! let data = z.data().unwrap();
//!
//! // Backpropagation
//! z.backward();
//! let grad = x.grad();
//! ```

pub mod dimension;
pub mod ops;

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

pub use dimension::{
    Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension, HasLarger, HasSmaller,
};

use crate::graph::{DType, GraphNode};

/// Gradient function trait for backpropagation
///
/// This trait is implemented by operations that can compute gradients.
pub trait GradFn: Send + Sync {
    /// Compute gradients with respect to inputs
    ///
    /// # Arguments
    /// * `grad_output` - Gradient flowing back from the output
    ///
    /// # Returns
    /// Gradients for each input tensor
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>>;

    /// Get the name of this gradient function (for debugging)
    fn name(&self) -> &'static str;
}

/// Internal data for a tensor (for autograd tracking)
///
/// This struct is used for gradient computation and caching.
/// Fields will be used in forward() and backward() implementations.
#[allow(dead_code)]
struct TensorData {
    /// Whether this tensor requires gradient computation
    requires_grad: bool,
    /// Stored gradient (populated after backward())
    grad: RefCell<Option<Rc<Tensor<DimDyn>>>>,
    /// Gradient function for backpropagation
    grad_fn: Option<Rc<dyn GradFn>>,
    /// Cached data after forward pass (if executed)
    cached_data: RefCell<Option<Vec<f32>>>,
}

/// A multi-dimensional tensor with lazy evaluation and automatic differentiation
///
/// # Type Parameters
///
/// * `D` - The dimension type, either static (`Dim<N>`) or dynamic (`DimDyn`)
///
/// # Features
///
/// - **Lazy Evaluation**: Operations build a computation graph without immediate execution
/// - **Static Dimensions**: Use `Tensor<Dim<N>>` for compile-time dimension checking
/// - **Dynamic Dimensions**: Use `Tensor<DimDyn>` for runtime-determined dimensions
/// - **Automatic Differentiation**: Track gradients with `.requires_grad()`
///
/// # Examples
///
/// ```ignore
/// use harp::tensor::{Tensor, Dim2};
///
/// // Static 2D tensor
/// let matrix = Tensor::<Dim2>::zeros([3, 4]);
///
/// // Dynamic dimension tensor
/// let dynamic = Tensor::<DimDyn>::zeros(&[3, 4, 5]);
/// ```
pub struct Tensor<D: Dimension = DimDyn> {
    /// The computation graph node (forms a DAG through `src` references)
    node: GraphNode,
    /// Shape of the tensor (cached from node.view for convenience)
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Autograd data (only allocated when requires_grad is true)
    autograd: Option<Rc<TensorData>>,
    /// Marker for dimension type
    _dim: PhantomData<D>,
}

impl<D: Dimension> Clone for Tensor<D> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            autograd: self.autograd.clone(),
            _dim: PhantomData,
        }
    }
}

impl<D: Dimension> Tensor<D> {
    /// Create a new tensor from a graph node
    fn from_node(node: GraphNode, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            node,
            shape,
            dtype,
            autograd: None,
            _dim: PhantomData,
        }
    }

    /// Create a new tensor from a graph node with autograd support
    #[allow(dead_code)]
    fn from_node_with_grad(
        node: GraphNode,
        shape: Vec<usize>,
        dtype: DType,
        grad_fn: Option<Rc<dyn GradFn>>,
    ) -> Self {
        let autograd = Some(Rc::new(TensorData {
            requires_grad: true,
            grad: RefCell::new(None),
            grad_fn,
            cached_data: RefCell::new(None),
        }));
        Self {
            node,
            shape,
            dtype,
            autograd,
            _dim: PhantomData,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if this tensor requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.autograd.is_some()
    }

    /// Enable gradient tracking for this tensor
    pub fn set_requires_grad(mut self, requires_grad: bool) -> Self {
        if requires_grad && self.autograd.is_none() {
            self.autograd = Some(Rc::new(TensorData {
                requires_grad: true,
                grad: RefCell::new(None),
                grad_fn: None,
                cached_data: RefCell::new(None),
            }));
        } else if !requires_grad {
            self.autograd = None;
        }
        self
    }

    /// Get a reference to the underlying graph node
    pub fn node(&self) -> &GraphNode {
        &self.node
    }

    /// Convert to a dynamic dimension tensor
    pub fn into_dyn(self) -> Tensor<DimDyn> {
        Tensor {
            node: self.node,
            shape: self.shape,
            dtype: self.dtype,
            autograd: self.autograd,
            _dim: PhantomData,
        }
    }
}

// Static dimension constructors
impl<const N: usize> Tensor<Dim<N>> {
    /// Create a tensor filled with zeros
    ///
    /// # Arguments
    /// * `shape` - The shape as a fixed-size array
    ///
    /// # Example
    ///
    /// ```ignore
    /// let zeros = Tensor::<Dim2>::zeros([3, 4]);
    /// ```
    pub fn zeros(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let node = GraphNode::zeros(shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: [usize; N]) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let node = GraphNode::ones(shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: [usize; N], value: f32) -> Self {
        let shape_vec: Vec<usize> = shape.to_vec();
        let node = GraphNode::full(value, shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create an input tensor (placeholder for data)
    ///
    /// # Arguments
    /// * `name` - Name for this input
    /// * `shape` - The shape as a fixed-size array
    pub fn input(name: &str, shape: [usize; N]) -> Self {
        use crate::graph::{GraphOp, shape::View};
        let shape_exprs: Vec<crate::graph::shape::Expr> = shape
            .iter()
            .map(|&s| crate::graph::shape::Expr::from(s as i64))
            .collect();
        let view = View::contiguous(shape_exprs);
        let node = GraphNode::new(
            DType::F32,
            GraphOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
        );
        Self::from_node(node, shape.to_vec(), DType::F32)
    }
}

// Dynamic dimension constructors
impl Tensor<DimDyn> {
    /// Create a tensor filled with zeros (dynamic shape)
    ///
    /// # Arguments
    /// * `shape` - The shape as a slice
    pub fn zeros_dyn(shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let node = GraphNode::zeros(shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create a tensor filled with ones (dynamic shape)
    pub fn ones_dyn(shape: &[usize]) -> Self {
        let shape_vec = shape.to_vec();
        let node = GraphNode::ones(shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create a tensor filled with a constant value (dynamic shape)
    pub fn full_dyn(shape: &[usize], value: f32) -> Self {
        let shape_vec = shape.to_vec();
        let node = GraphNode::full(value, shape_vec.clone());
        Self::from_node(node, shape_vec, DType::F32)
    }

    /// Create an input tensor (dynamic shape)
    pub fn input_dyn(name: &str, shape: &[usize]) -> Self {
        use crate::graph::{GraphOp, shape::View};
        let shape_exprs: Vec<crate::graph::shape::Expr> = shape
            .iter()
            .map(|&s| crate::graph::shape::Expr::from(s as i64))
            .collect();
        let view = View::contiguous(shape_exprs);
        let node = GraphNode::new(
            DType::F32,
            GraphOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
        );
        Self::from_node(node, shape.to_vec(), DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros_static() {
        let t = Tensor::<Dim2>::zeros([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 12);
    }

    #[test]
    fn test_tensor_ones_static() {
        let t = Tensor::<Dim3>::ones([2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_tensor_full_static() {
        let t = Tensor::<Dim1>::full([10], 3.14);
        assert_eq!(t.shape(), &[10]);
        assert_eq!(t.ndim(), 1);
    }

    #[test]
    fn test_tensor_zeros_dynamic() {
        let t = Tensor::<DimDyn>::zeros_dyn(&[3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 60);
    }

    #[test]
    fn test_tensor_input() {
        let t = Tensor::<Dim2>::input("x", [10, 20]);
        assert_eq!(t.shape(), &[10, 20]);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_into_dyn() {
        let static_tensor = Tensor::<Dim2>::zeros([3, 4]);
        let dyn_tensor = static_tensor.into_dyn();
        assert_eq!(dyn_tensor.shape(), &[3, 4]);
    }
}
