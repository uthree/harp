//! Tensor structure with lazy evaluation
//!
//! The `Tensor<D>` type wraps a computation graph node and provides
//! type-safe tensor operations with compile-time dimension checking.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

use crate::ast::{DType, TensorDType};
use crate::backend::Buffer;
use crate::graph::{self, Expr, GraphNode};

use super::dim::{Dimension, Dyn};

// ============================================================================
// Tensor Structure
// ============================================================================

#[derive(Clone)]
pub struct Tensor<D: Dimension, T: TensorDType = f32> {
    pub(crate) inner: Rc<TensorInner<D, T>>,
}

pub(crate) struct TensorInner<D: Dimension, T: TensorDType = f32> {
    /// The underlying computation graph node.
    pub(crate) graph: GraphNode,

    /// Realized buffer (None if not yet realized).
    /// Uses RefCell for interior mutability since Tensor is shared via Rc.
    pub(crate) buffer: RefCell<Option<Box<dyn Buffer>>>,

    /// Whether this tensor requires gradient computation.
    /// When true, gradients will be computed during the backward pass.
    pub(crate) requires_grad: Cell<bool>,

    /// Stored gradient from backward pass.
    /// This is `Some` after backward() is called on a downstream tensor.
    pub(crate) grad: RefCell<Option<Tensor<D, T>>>,

    /// Phantom marker for the dimension type.
    pub(crate) _dim: PhantomData<D>,

    /// Phantom marker for the data type.
    pub(crate) _dtype: PhantomData<T>,
}

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a tensor from a GraphNode.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the GraphNode's dimensionality doesn't match `D`.
    pub fn from_graph(graph: GraphNode) -> Self {
        debug_assert!(
            D::check_shape(
                &graph
                    .shape()
                    .iter()
                    .filter_map(|e| e.as_usize())
                    .collect::<Vec<_>>()
            ) || D::NDIM == usize::MAX, // Dyn accepts any shape
            "GraphNode shape doesn't match dimension type D"
        );

        Self {
            inner: Rc::new(TensorInner {
                graph,
                buffer: RefCell::new(None),
                requires_grad: Cell::new(false),
                grad: RefCell::new(None),
                _dim: PhantomData,
                _dtype: PhantomData,
            }),
        }
    }

    /// Create an input tensor placeholder.
    ///
    /// This creates a tensor that will receive data at execution time.
    /// The data type is determined by the type parameter `T`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use eclat::tensor::{Tensor, D2};
    ///
    /// let x: Tensor<D2, f32> = Tensor::input([32, 64]);
    /// assert_eq!(x.shape(), vec![32, 64]);
    /// ```
    pub fn input<const N: usize>(shape: [usize; N]) -> Self
    where
        D: Dimension,
    {
        assert_eq!(
            N,
            D::NDIM,
            "Shape length {} doesn't match dimension {}",
            N,
            D::NDIM
        );

        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = graph::input(expr_shape, T::DTYPE);
        Self::from_graph(graph)
    }

    /// Create a named input tensor placeholder.
    ///
    /// Named inputs are useful for debugging and identifying tensors
    /// in the computation graph. The data type is determined by `T`.
    pub fn named_input<const N: usize>(name: &str, shape: [usize; N]) -> Self
    where
        D: Dimension,
    {
        assert_eq!(
            N,
            D::NDIM,
            "Shape length {} doesn't match dimension {}",
            N,
            D::NDIM
        );

        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = graph::named_input(name, expr_shape, T::DTYPE);
        Self::from_graph(graph)
    }

    /// Create a tensor filled with zeros.
    /// The data type is determined by `T`.
    pub fn zeros<const N: usize>(shape: [usize; N]) -> Self
    where
        D: Dimension,
    {
        assert_eq!(
            N,
            D::NDIM,
            "Shape length {} doesn't match dimension {}",
            N,
            D::NDIM
        );

        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = graph::zeros(expr_shape, T::DTYPE);
        Self::from_graph(graph)
    }

    /// Create a tensor filled with ones.
    /// The data type is determined by `T`.
    pub fn ones<const N: usize>(shape: [usize; N]) -> Self
    where
        D: Dimension,
    {
        assert_eq!(
            N,
            D::NDIM,
            "Shape length {} doesn't match dimension {}",
            N,
            D::NDIM
        );

        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = graph::ones(expr_shape, T::DTYPE);
        Self::from_graph(graph)
    }

    // ========================================================================
    // Dynamic Dimension Constructors
    // ========================================================================

    /// Create a tensor with dynamic dimensions from shape and data.
    ///
    /// This is primarily used for neural network parameters where the
    /// dimension count is determined at runtime.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor
    /// * `data` - The initial data (flattened)
    ///
    /// # Panics
    /// Panics if `data.len()` doesn't match the product of `shape`.
    pub fn from_shape_data(shape: &[usize], data: &[f32]) -> Tensor<Dyn, f32> {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (numel={})",
            data.len(),
            shape,
            numel
        );

        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = graph::input(expr_shape, DType::F32);
        Tensor::<Dyn, f32>::from_graph(graph)
    }

    /// Create an input tensor with dynamic dimensions.
    ///
    /// Unlike `input()` which requires compile-time dimension count,
    /// this method accepts a runtime slice.
    pub fn dyn_input(shape: &[usize]) -> Tensor<Dyn, f32> {
        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = graph::input(expr_shape, DType::F32);
        Tensor::<Dyn, f32>::from_graph(graph)
    }

    /// Create a zeros tensor with dynamic dimensions.
    pub fn dyn_zeros(shape: &[usize]) -> Tensor<Dyn, f32> {
        let expr_shape: Vec<Expr> = shape.iter().map(|&s| Expr::Const(s as i64)).collect();
        let graph = graph::zeros(expr_shape, DType::F32);
        Tensor::<Dyn, f32>::from_graph(graph)
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /// Get the tensor's shape as a vector of sizes.
    ///
    /// Note: For tensors with symbolic dimensions, this may fail.
    pub fn shape(&self) -> Vec<usize> {
        self.inner
            .graph
            .shape()
            .iter()
            .map(|e| {
                e.as_usize()
                    .expect("Cannot get concrete shape for symbolic dimension")
            })
            .collect()
    }

    /// Get the tensor's shape as symbolic expressions.
    pub fn shape_expr(&self) -> Vec<Expr> {
        self.inner.graph.shape().clone()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.inner.graph.ndim()
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.inner.graph.dtype().clone()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Get a reference to the underlying GraphNode.
    pub fn graph(&self) -> &GraphNode {
        &self.inner.graph
    }

    /// Set a name for this tensor (for debugging).
    pub fn with_name(self, name: &str) -> Self {
        let graph = self.inner.graph.clone().with_name(name);
        Self::from_graph(graph)
    }

    // ========================================================================
    // Conversion
    // ========================================================================

    /// Convert to a dynamic dimension tensor.
    ///
    /// This is useful when the dimension is not known at compile time.
    pub fn into_dyn(self) -> Tensor<super::dim::Dyn, T> {
        Tensor::from_graph(self.inner.graph.clone())
    }
}

impl<T: TensorDType> Tensor<super::dim::Dyn, T> {
    /// Try to convert a dynamic dimension tensor to a static dimension tensor.
    ///
    /// Returns `Some` if the actual number of dimensions matches `D::NDIM`,
    /// otherwise returns `None`.
    ///
    /// # Example
    /// ```ignore
    /// let dyn_tensor: Tensor<Dyn, f32> = Tensor::dyn_input(&[1, 3, 32, 32]);
    /// let d4_tensor: Option<Tensor<D4, f32>> = dyn_tensor.try_into_static();
    /// ```
    pub fn try_into_static<D: Dimension>(self) -> Option<Tensor<D, T>> {
        if self.ndim() == D::NDIM {
            Some(Tensor::from_graph(self.inner.graph.clone()))
        } else {
            None
        }
    }

    /// Convert a dynamic dimension tensor to a static dimension tensor.
    ///
    /// # Panics
    /// Panics if the actual number of dimensions doesn't match `D::NDIM`.
    pub fn into_static<D: Dimension>(self) -> Tensor<D, T> {
        let ndim = self.ndim();
        self.try_into_static()
            .unwrap_or_else(|| panic!("Expected {} dimensions, got {}", D::NDIM, ndim))
    }
}

// ============================================================================
// Debug Implementation
// ============================================================================

impl<D: Dimension, T: TensorDType> std::fmt::Debug for Tensor<D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("ndim", &D::NDIM)
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

// ============================================================================
// Scalar Tensor (D0) Specific Methods
// ============================================================================

impl<T: TensorDType> Tensor<super::dim::D0, T> {
    /// Create a scalar tensor from a constant value.
    /// The data type is determined by `T`.
    pub fn scalar(_value: T) -> Self {
        let graph = graph::scalar(T::DTYPE);
        // Note: The actual value will be set during realize()
        // For now, create a constant node
        Self::from_graph(graph)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dim::{D0, D1, D2, D3};

    #[test]
    fn test_tensor_input() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        assert_eq!(x.shape(), vec![32, 64]);
        assert_eq!(x.ndim(), 2);
        assert_eq!(x.dtype(), DType::F32);
    }

    #[test]
    fn test_tensor_named_input() {
        let x: Tensor<D2, f32> = Tensor::named_input("input", [32, 64]);
        assert_eq!(x.shape(), vec![32, 64]);
    }

    #[test]
    fn test_tensor_zeros() {
        let x: Tensor<D3, f32> = Tensor::zeros([2, 3, 4]);
        assert_eq!(x.shape(), vec![2, 3, 4]);
        assert_eq!(x.ndim(), 3);
    }

    #[test]
    fn test_tensor_ones() {
        let x: Tensor<D1, f64> = Tensor::ones([100]);
        assert_eq!(x.shape(), vec![100]);
        assert_eq!(x.dtype(), DType::F64);
    }

    #[test]
    fn test_tensor_scalar() {
        let x: Tensor<D0, f32> = Tensor::scalar(2.5);
        assert_eq!(x.ndim(), 0);
        assert_eq!(x.shape(), Vec::<usize>::new());
    }

    #[test]
    fn test_tensor_numel() {
        let x: Tensor<D3, f32> = Tensor::input([2, 3, 4]);
        assert_eq!(x.numel(), 24);
    }

    #[test]
    fn test_tensor_into_dyn() {
        let x: Tensor<D2, f32> = Tensor::input([32, 64]);
        let x_dyn = x.into_dyn();
        assert_eq!(x_dyn.shape(), vec![32, 64]);
    }

    #[test]
    #[should_panic(expected = "Shape length 2 doesn't match dimension 3")]
    fn test_tensor_dimension_mismatch() {
        // This should panic because we're trying to create a D3 tensor with a 2D shape
        let _x: Tensor<D3, f32> = Tensor::input([32, 64]);
    }
}
