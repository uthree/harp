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
//! ## primops (Primitive Operations)
//!
//! Minimal set of operations from which all others are composed:
//!
//! - **Initialization**: Const, Rand
//! - **Binary**: Add, Mul, Max, Idiv
//! - **Unary**: Neg, Recip, Sqrt, Log2, Exp2, Sin
//! - **Reduce**: Reduce(Add), Reduce(Mul), Reduce(Max)
//! - **Movement**: Squeeze, Unsqueeze, Repeat, Reshape, Contiguous
//!
//! ## hlops (High-Level Operations)
//!
//! Composed from primops for convenience:
//!
//! - **Arithmetic**: Sub = Add(a, Neg(b)), Div = Mul(a, Recip(b))
//! - **Transcendental**: Exp, Ln, Cos, Tan, Pow
//! - **Activation**: ReLU, Sigmoid, Tanh, GELU, SiLU
//! - **Reduction**: Mean, Var, Std, Softmax, LogSoftmax
//! - **Linear Algebra**: MatMul, Dot, Outer
//!
//! # Examples
//!
//! ```ignore
//! use harp::tensor::{Tensor, Dim2};
//!
//! // Create a 2D tensor with gradient tracking
//! let x = Tensor::<Dim2>::zeros([3, 4]).set_requires_grad(true);
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
pub mod forward;
pub mod hlops;
pub mod primops;

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

pub use dimension::{Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
pub use forward::ForwardError;

use crate::graph::{DType, GraphNode};

/// Gradient function trait for backpropagation
///
/// This trait is implemented by operations that can compute gradients.
pub trait GradFn {
    /// Compute gradients with respect to inputs
    ///
    /// # Arguments
    /// * `grad_output` - Gradient flowing back from the output
    ///
    /// # Returns
    /// Gradients for each input tensor
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>>;

    /// Get the input tensors that this gradient function operates on
    ///
    /// This is used to propagate gradients back through the computation graph.
    fn inputs(&self) -> Vec<Tensor<DimDyn>>;

    /// Get the name of this gradient function (for debugging)
    fn name(&self) -> &'static str;
}

/// Internal data for a tensor (for autograd tracking)
///
/// This struct is used for gradient computation and caching.
/// Fields will be used in forward() and backward() implementations.
pub(crate) struct TensorData {
    /// Whether this tensor requires gradient computation
    #[allow(dead_code)]
    pub(crate) requires_grad: bool,
    /// Stored gradient (populated after backward())
    pub(crate) grad: RefCell<Option<Rc<Tensor<DimDyn>>>>,
    /// Gradient function for backpropagation
    pub(crate) grad_fn: Option<Rc<dyn GradFn>>,
    /// Cached data after forward pass (if executed)
    pub(crate) cached_data: RefCell<Option<Vec<f32>>>,
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
/// - **Automatic Differentiation**: Track gradients with `.set_requires_grad(true)`
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
/// let dynamic = Tensor::<DimDyn>::zeros_dyn(&[3, 4, 5]);
/// ```
pub struct Tensor<D: Dimension = DimDyn> {
    /// The computation graph node (forms a DAG through `src` references)
    pub(crate) node: GraphNode,
    /// Shape of the tensor (cached from node.view for convenience)
    pub(crate) shape: Vec<usize>,
    /// Data type
    pub(crate) dtype: DType,
    /// Autograd data (only allocated when requires_grad is true)
    pub(crate) autograd: Option<Rc<TensorData>>,
    /// Marker for dimension type
    pub(crate) _dim: PhantomData<D>,
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

// ============================================================================
// Backward propagation
// ============================================================================

impl<D: Dimension> Tensor<D> {
    /// Perform backward propagation from this tensor
    ///
    /// Computes gradients for all tensors in the computation graph that
    /// have `requires_grad = true`.
    ///
    /// # Panics
    ///
    /// Panics if this tensor does not require gradients.
    pub fn backward(&self) {
        if self.autograd.is_none() {
            panic!("backward() called on tensor that doesn't require gradients");
        }

        // Create initial gradient of ones with same shape
        let initial_grad = Tensor::<DimDyn>::ones_dyn(self.shape());
        self.backward_with(initial_grad);
    }

    /// Perform backward propagation with a custom initial gradient
    pub fn backward_with(&self, grad_output: Tensor<DimDyn>) {
        if let Some(ref autograd) = self.autograd {
            // Accumulate gradient
            {
                let mut grad = autograd.grad.borrow_mut();
                if let Some(existing) = grad.take() {
                    // Add to existing gradient
                    let new_grad = &(*existing) + &grad_output;
                    *grad = Some(Rc::new(new_grad));
                } else {
                    *grad = Some(Rc::new(grad_output.clone()));
                }
            }

            // Propagate to inputs via grad_fn
            if let Some(ref grad_fn) = autograd.grad_fn {
                let input_grads = grad_fn.backward(&grad_output);
                let inputs = grad_fn.inputs();

                // Propagate gradients to each input tensor
                for (input, grad) in inputs.into_iter().zip(input_grads.into_iter()) {
                    if input.requires_grad() {
                        input.backward_with(grad);
                    }
                }
            }
        }
    }

    /// Get the accumulated gradient for this tensor
    ///
    /// Returns None if backward() hasn't been called or if this tensor
    /// doesn't require gradients.
    pub fn grad(&self) -> Option<Tensor<DimDyn>> {
        self.autograd
            .as_ref()
            .and_then(|ag| ag.grad.borrow().as_ref().map(|g| (**g).clone()))
    }

    /// Reset the gradient to None
    pub fn zero_grad(&self) {
        if let Some(ref autograd) = self.autograd {
            *autograd.grad.borrow_mut() = None;
        }
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    pub fn detach(&self) -> Tensor<D> {
        Tensor {
            node: self.node.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            autograd: None,
            _dim: PhantomData,
        }
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
        let t = Tensor::<Dim1>::full([10], 2.5);
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

    #[test]
    fn test_set_requires_grad() {
        let t = Tensor::<Dim2>::ones([2, 2]).set_requires_grad(true);
        assert!(t.requires_grad());
    }

    #[test]
    fn test_detach() {
        let t = Tensor::<Dim2>::ones([2, 2]).set_requires_grad(true);
        let detached = t.detach();
        assert!(!detached.requires_grad());
    }

    #[test]
    fn test_rand() {
        let t = Tensor::<Dim2>::rand([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }
}
