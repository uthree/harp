//! Tensor structure with lazy evaluation
//!
//! The `Tensor<D>` type wraps a computation graph node and provides
//! type-safe tensor operations with compile-time dimension checking.

use std::marker::PhantomData;
use std::rc::Rc;

use crate::ast::DType;
use crate::graph::{self, Expr, GraphNode};

use super::dim::Dimension;

// ============================================================================
// Tensor Structure
// ============================================================================

/// A tensor with statically-checked dimensions.
///
/// `Tensor<D>` wraps a `GraphNode` computation graph and provides:
/// - Compile-time dimension checking via the `D` type parameter
/// - Lazy evaluation: operations build a graph, `realize()` executes it
/// - Type-safe operations that preserve dimension information
///
/// # Type Parameters
///
/// - `D`: A type implementing `Dimension` that represents the tensor's dimensionality
///
/// # Examples
///
/// ```ignore
/// use eclat::tensor::{Tensor, D2, D1};
/// use eclat::ast::DType;
///
/// // Create a 2D input tensor
/// let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
///
/// // Operations preserve type information
/// let y: Tensor<D2> = &x + &x;  // D2 + D2 = D2
/// let z: Tensor<D1> = y.sum(1); // sum over axis 1: D2 -> D1
/// ```
#[derive(Clone)]
pub struct Tensor<D: Dimension> {
    pub(crate) inner: Rc<TensorInner<D>>,
}

/// Internal tensor data.
pub(crate) struct TensorInner<D: Dimension> {
    /// The underlying computation graph node.
    pub(crate) graph: GraphNode,

    /// Phantom marker for the dimension type.
    pub(crate) _dim: PhantomData<D>,
}

impl<D: Dimension> Tensor<D> {
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
                _dim: PhantomData,
            }),
        }
    }

    /// Create an input tensor placeholder.
    ///
    /// This creates a tensor that will receive data at execution time.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use eclat::tensor::{Tensor, D2};
    /// use eclat::ast::DType;
    ///
    /// let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
    /// assert_eq!(x.shape(), vec![32, 64]);
    /// ```
    pub fn input<const N: usize>(shape: [usize; N], dtype: DType) -> Self
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
        let graph = graph::input(expr_shape, dtype);
        Self::from_graph(graph)
    }

    /// Create a named input tensor placeholder.
    ///
    /// Named inputs are useful for debugging and identifying tensors
    /// in the computation graph.
    pub fn named_input<const N: usize>(name: &str, shape: [usize; N], dtype: DType) -> Self
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
        let graph = graph::named_input(name, expr_shape, dtype);
        Self::from_graph(graph)
    }

    /// Create a tensor filled with zeros.
    pub fn zeros<const N: usize>(shape: [usize; N], dtype: DType) -> Self
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
        let graph = graph::zeros(expr_shape, dtype);
        Self::from_graph(graph)
    }

    /// Create a tensor filled with ones.
    pub fn ones<const N: usize>(shape: [usize; N], dtype: DType) -> Self
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
        let graph = graph::ones(expr_shape, dtype);
        Self::from_graph(graph)
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
    pub fn into_dyn(self) -> Tensor<super::dim::Dyn> {
        Tensor::from_graph(self.inner.graph.clone())
    }
}

// ============================================================================
// Debug Implementation
// ============================================================================

impl<D: Dimension> std::fmt::Debug for Tensor<D> {
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

impl Tensor<super::dim::D0> {
    /// Create a scalar tensor from a constant value.
    pub fn scalar(_value: f32, dtype: DType) -> Self {
        let graph = graph::scalar(dtype);
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
        let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        assert_eq!(x.shape(), vec![32, 64]);
        assert_eq!(x.ndim(), 2);
        assert_eq!(x.dtype(), DType::F32);
    }

    #[test]
    fn test_tensor_named_input() {
        let x: Tensor<D2> = Tensor::named_input("input", [32, 64], DType::F32);
        assert_eq!(x.shape(), vec![32, 64]);
    }

    #[test]
    fn test_tensor_zeros() {
        let x: Tensor<D3> = Tensor::zeros([2, 3, 4], DType::F32);
        assert_eq!(x.shape(), vec![2, 3, 4]);
        assert_eq!(x.ndim(), 3);
    }

    #[test]
    fn test_tensor_ones() {
        let x: Tensor<D1> = Tensor::ones([100], DType::F64);
        assert_eq!(x.shape(), vec![100]);
        assert_eq!(x.dtype(), DType::F64);
    }

    #[test]
    fn test_tensor_scalar() {
        let x: Tensor<D0> = Tensor::scalar(3.14, DType::F32);
        assert_eq!(x.ndim(), 0);
        assert_eq!(x.shape(), Vec::<usize>::new());
    }

    #[test]
    fn test_tensor_numel() {
        let x: Tensor<D3> = Tensor::input([2, 3, 4], DType::F32);
        assert_eq!(x.numel(), 24);
    }

    #[test]
    fn test_tensor_into_dyn() {
        let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
        let x_dyn = x.into_dyn();
        assert_eq!(x_dyn.shape(), vec![32, 64]);
    }

    #[test]
    #[should_panic(expected = "Shape length 2 doesn't match dimension 3")]
    fn test_tensor_dimension_mismatch() {
        // This should panic because we're trying to create a D3 tensor with a 2D shape
        let _x: Tensor<D3> = Tensor::input([32, 64], DType::F32);
    }
}
