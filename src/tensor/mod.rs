//! Unified Tensor type for Harp
//!
//! This module provides a high-level `Tensor<D>` type that combines:
//! - Lazy evaluation via computation graphs
//! - Static dimension checking via const generics
//! - Automatic differentiation (autograd)
//! - Eager fusion for operation optimization
//!
//! # Design Philosophy
//!
//! The Tensor type follows the design philosophy of tinygrad/micrograd:
//! minimal primitives combined to create complex functionality.
//!
//! ## Architecture
//!
//! Tensor holds TensorInner which contains TensorOp.
//! Input tensors are embedded directly in TensorOp variants.
//!
//! ## primops (Primitive Operations)
//!
//! Minimal set of operations from which all others are composed:
//!
//! - **Initialization**: Const, Rand, Arange
//! - **Binary**: Add, Mul, Max, Idiv, Rem
//! - **Unary**: Neg, Recip, Sqrt, Log2, Exp2, Sin, Floor
//! - **Reduce**: Reduce(Sum), Reduce(Prod), Reduce(Max)
//! - **Movement**: View, Contiguous, Pad, Slice
//! - **Special**: Clone (explicit branch point), Cast
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
//! ## Eager Fusion
//!
//! Operations are fused at call time using the unified Compute variant.
//!
//! ## Ownership-based Fusion Control
//!
//! Operations consume `self` (move semantics). For branching, use explicit `fork()`:
//! ```ignore
//! let a = x + y;           // x, y consumed
//! let b = a.sum();         // a consumed → fusion OK
//!
//! // For branching:
//! let a = x + y;
//! let a2 = a.fork();       // Clone op added to graph
//! let b = a.sum();         // a consumed → fusion OK
//! let c = a2 * 2.0;        // a2 is separate path
//! ```
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
//! z.contiguous();
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
pub mod fusion;
pub mod hlops;
pub mod lowerer;
pub mod ops;
pub mod primops;
pub mod shape;

use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

pub use dimension::{Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
pub use forward::ForwardError;
pub use ops::{ElementwiseOp, ReduceOp, TensorOp, TensorRef};
pub use shape::{Expr, View};

use crate::ast::DType;

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

    /// Get the input tensors that this gradient function operates on
    ///
    /// This is used to propagate gradients back through the computation graph.
    fn inputs(&self) -> Vec<Tensor<DimDyn>>;

    /// Get the name of this gradient function (for debugging)
    fn name(&self) -> &'static str;
}

/// Autograd metadata for gradient tracking
///
/// The presence of this struct indicates that gradient tracking is enabled.
/// Use `tensor.requires_grad()` (which checks `autograd.is_some()`) to check status.
pub struct AutogradMeta {
    /// Stored gradient (populated after backward())
    pub(crate) grad: RwLock<Option<Arc<Tensor<DimDyn>>>>,
    /// Gradient function for backpropagation
    pub(crate) grad_fn: Option<Arc<dyn GradFn>>,
}

// ============================================================================
// TensorInner: Internal tensor data
// ============================================================================

/// Internal tensor data (unified from old Tensor + TensorNode)
///
/// This structure is reference-counted via Arc for efficient sharing.
/// Input tensors are embedded in TensorOp variants.
pub struct TensorInner {
    /// The operation that produces this tensor (includes inputs)
    pub(crate) op: TensorOp,
    /// Memory layout information
    pub(crate) view: View,
    /// Shape of the tensor
    pub(crate) shape: Vec<usize>,
    /// Data type
    pub(crate) dtype: DType,
    /// Optional name for debugging
    #[allow(dead_code)]
    pub(crate) name: Option<String>,
    /// Autograd metadata (only allocated when requires_grad is true)
    pub(crate) autograd: Option<AutogradMeta>,
    /// Executed buffer data (populated after contiguous())
    pub(crate) buffer: RwLock<Option<Vec<f32>>>,
}

impl TensorInner {
    /// Create a new tensor inner
    pub fn new(op: TensorOp, view: View, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            op,
            view,
            shape,
            dtype,
            name: None,
            autograd: None,
            buffer: RwLock::new(None),
        }
    }

    /// Create a new tensor inner with a name
    pub fn new_named(
        op: TensorOp,
        view: View,
        shape: Vec<usize>,
        dtype: DType,
        name: impl Into<String>,
    ) -> Self {
        Self {
            op,
            view,
            shape,
            dtype,
            name: Some(name.into()),
            autograd: None,
            buffer: RwLock::new(None),
        }
    }
}

// ============================================================================
// Tensor: The main tensor type
// ============================================================================

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
/// - **Eager Fusion**: Operations are fused at call time for optimization
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
    /// Internal tensor data (reference counted for efficient sharing)
    pub(crate) inner: Arc<TensorInner>,
    /// Marker for dimension type
    pub(crate) _dim: PhantomData<D>,
}

// Implement Send and Sync for Tensor
unsafe impl<D: Dimension> Send for Tensor<D> {}
unsafe impl<D: Dimension> Sync for Tensor<D> {}

impl<D: Dimension> Clone for Tensor<D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _dim: PhantomData,
        }
    }
}

impl<D: Dimension> Tensor<D> {
    /// Create an explicit branch point in the computation graph (Clone operation)
    ///
    /// Use this when you need to use the same tensor in multiple computation paths.
    /// This creates a Clone operation in the graph that will copy the buffer at execution time.
    ///
    /// # Example
    /// ```ignore
    /// let a = x + y;
    /// let a2 = a.fork();       // Clone op added to graph
    /// let b = a.sum();         // a consumed → fusion OK
    /// let c = a2 * 2.0;        // a2 is separate path
    /// ```
    pub fn fork(&self) -> Tensor<D> {
        let input = Arc::new(self.clone().into_dyn());
        let inner = TensorInner::new(
            TensorOp::Clone { input },
            self.inner.view.clone(),
            self.inner.shape.clone(),
            self.inner.dtype.clone(),
        );
        Tensor {
            inner: Arc::new(inner),
            _dim: PhantomData,
        }
    }
}

impl<D: Dimension> Tensor<D> {
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    /// Get the data type of the tensor
    pub fn dtype(&self) -> &DType {
        &self.inner.dtype
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.inner.shape.iter().product()
    }

    /// Check if this tensor requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.inner.autograd.is_some()
    }

    /// Enable gradient tracking for this tensor
    ///
    /// Note: This creates a new tensor with gradient tracking enabled.
    /// The original tensor is consumed.
    pub fn set_requires_grad(self, requires_grad: bool) -> Self {
        if requires_grad && self.inner.autograd.is_none() {
            // Create new inner with autograd
            let inner = TensorInner {
                op: self.inner.op.clone(),
                view: self.inner.view.clone(),
                shape: self.inner.shape.clone(),
                dtype: self.inner.dtype.clone(),
                name: self.inner.name.clone(),
                autograd: Some(AutogradMeta {
                    grad: RwLock::new(None),
                    grad_fn: None,
                }),
                buffer: RwLock::new(self.inner.buffer.read().unwrap().clone()),
            };
            Tensor {
                inner: Arc::new(inner),
                _dim: PhantomData,
            }
        } else if !requires_grad && self.inner.autograd.is_some() {
            // Create new inner without autograd
            let inner = TensorInner {
                op: self.inner.op.clone(),
                view: self.inner.view.clone(),
                shape: self.inner.shape.clone(),
                dtype: self.inner.dtype.clone(),
                name: self.inner.name.clone(),
                autograd: None,
                buffer: RwLock::new(self.inner.buffer.read().unwrap().clone()),
            };
            Tensor {
                inner: Arc::new(inner),
                _dim: PhantomData,
            }
        } else {
            self
        }
    }

    /// Convert to a dynamic dimension tensor
    pub fn into_dyn(self) -> Tensor<DimDyn> {
        Tensor {
            inner: self.inner,
            _dim: PhantomData,
        }
    }

    /// Get the view of this tensor's memory layout
    pub fn view(&self) -> &View {
        &self.inner.view
    }

    /// Get the operation that produces this tensor
    pub fn op(&self) -> &TensorOp {
        &self.inner.op
    }

    /// Check if this tensor has been executed (buffer is populated)
    pub fn is_executed(&self) -> bool {
        self.inner.buffer.read().unwrap().is_some()
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
        if self.inner.autograd.is_none() {
            panic!("backward() called on tensor that doesn't require gradients");
        }

        // Create initial gradient of ones with same shape
        let initial_grad = Tensor::<DimDyn>::ones_dyn(self.shape());
        self.backward_with(initial_grad);
    }

    /// Perform backward propagation with a custom initial gradient
    pub fn backward_with(&self, grad_output: Tensor<DimDyn>) {
        if let Some(ref autograd) = self.inner.autograd {
            // Accumulate gradient
            {
                let mut grad = autograd.grad.write().unwrap();
                if let Some(existing) = grad.take() {
                    // Add to existing gradient
                    let new_grad = &(*existing) + &grad_output;
                    *grad = Some(Arc::new(new_grad));
                } else {
                    *grad = Some(Arc::new(grad_output.clone()));
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
        self.inner
            .autograd
            .as_ref()
            .and_then(|ag| ag.grad.read().unwrap().as_ref().map(|g| (**g).clone()))
    }

    /// Reset the gradient to None
    pub fn zero_grad(&self) {
        if let Some(ref autograd) = self.inner.autograd {
            *autograd.grad.write().unwrap() = None;
        }
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    pub fn detach(&self) -> Tensor<D> {
        let inner = TensorInner {
            op: self.inner.op.clone(),
            view: self.inner.view.clone(),
            shape: self.inner.shape.clone(),
            dtype: self.inner.dtype.clone(),
            name: self.inner.name.clone(),
            autograd: None,
            buffer: RwLock::new(self.inner.buffer.read().unwrap().clone()),
        };
        Tensor {
            inner: Arc::new(inner),
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

    #[test]
    fn test_fork() {
        let t = Tensor::<Dim2>::ones([2, 3]);
        let forked = t.fork();

        // fork() creates a new tensor with the same shape
        assert_eq!(forked.shape(), t.shape());
        assert_eq!(forked.dtype(), t.dtype());

        // fork() does not preserve gradient tracking
        let t_grad = Tensor::<Dim2>::ones([2, 3]).set_requires_grad(true);
        let forked_grad = t_grad.fork();
        assert!(!forked_grad.requires_grad());

        // The forked tensor should have a Clone operation
        assert!(matches!(forked.inner.op, TensorOp::Clone { .. }));
    }
}
