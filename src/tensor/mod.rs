//! Unified Tensor type for Harp
//!
//! This module provides a high-level `Tensor<f32, D>` type that combines:
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
//! let x = Tensor::<f32, Dim2>::zeros([3, 4]).set_requires_grad(true);
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
pub mod dtype;
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
pub use dtype::{
    FloatDType, IntegerDType, NumericDType, SignedIntDType, TensorDType, UnsignedIntDType,
};
pub use forward::ForwardError;
pub use ops::{ElementwiseOp, ReduceOp, TensorOp, TensorRef};
pub use primops::{Exp2, Floor, Log2, Recip, Sin, Sqrt};
pub use shape::{Expr, View};

// ============================================================================
// Tensor type aliases (similar to ndarray's Array0, Array1, etc.)
// ============================================================================

/// 0-dimensional tensor (scalar)
pub type Tensor0 = Tensor<f32, Dim0>;
/// 1-dimensional tensor (vector)
pub type Tensor1 = Tensor<f32, Dim1>;
/// 2-dimensional tensor (matrix)
pub type Tensor2 = Tensor<f32, Dim2>;
/// 3-dimensional tensor
pub type Tensor3 = Tensor<f32, Dim3>;
/// 4-dimensional tensor (common for batched images: NCHW)
pub type Tensor4 = Tensor<f32, Dim4>;
/// 5-dimensional tensor
pub type Tensor5 = Tensor<f32, Dim5>;
/// 6-dimensional tensor
pub type Tensor6 = Tensor<f32, Dim6>;
/// Dynamic-dimensional tensor
pub type TensorDyn = Tensor<f32, DimDyn>;

use crate::ast::DType;

// ============================================================================
// GradFn trait - Generic over FloatDType
// ============================================================================

/// Gradient function trait for backpropagation
///
/// This trait is implemented by operations that can compute gradients.
/// Generic over T to support different floating-point types (f32, f64, future f16/bf16).
pub trait GradFn<T: FloatDType>: Send + Sync {
    /// Compute gradients with respect to inputs
    ///
    /// # Arguments
    /// * `grad_output` - Gradient flowing back from the output
    ///
    /// # Returns
    /// Gradients for each input tensor
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>>;

    /// Get the input tensors that this gradient function operates on
    ///
    /// This is used to propagate gradients back through the computation graph.
    fn inputs(&self) -> Vec<Tensor<T, DimDyn>>;

    /// Get the name of this gradient function (for debugging)
    fn name(&self) -> &'static str;
}

// ============================================================================
// AutogradMeta - Generic over FloatDType
// ============================================================================

/// Autograd metadata for gradient tracking (generic over T)
///
/// The presence of this struct indicates that gradient tracking is enabled.
pub struct AutogradMeta<T: FloatDType> {
    /// Stored gradient (populated after backward())
    pub(crate) grad: RwLock<Option<Arc<Tensor<T, DimDyn>>>>,
    /// Gradient function for backpropagation
    pub(crate) grad_fn: Option<Arc<dyn GradFn<T>>>,
}

// ============================================================================
// AutogradStorage - Type-erased storage for TensorInner
// ============================================================================

/// Type-erased autograd storage for TensorInner
///
/// Uses enum dispatch to support different FloatDType without dynamic dispatch overhead.
pub enum AutogradStorage {
    /// f32 autograd metadata
    F32(AutogradMeta<f32>),
    /// f64 autograd metadata
    F64(AutogradMeta<f64>),
    // Future: F16(AutogradMeta<f16>), BF16(AutogradMeta<bf16>)
}

impl AutogradStorage {
    /// Create new f32 autograd storage
    pub fn new_f32() -> Self {
        Self::F32(AutogradMeta {
            grad: RwLock::new(None),
            grad_fn: None,
        })
    }

    /// Create new f64 autograd storage
    pub fn new_f64() -> Self {
        Self::F64(AutogradMeta {
            grad: RwLock::new(None),
            grad_fn: None,
        })
    }

    /// Create new f32 autograd storage with gradient function
    pub fn new_f32_with_grad_fn(grad_fn: Arc<dyn GradFn<f32>>) -> Self {
        Self::F32(AutogradMeta {
            grad: RwLock::new(None),
            grad_fn: Some(grad_fn),
        })
    }

    /// Create new f64 autograd storage with gradient function
    pub fn new_f64_with_grad_fn(grad_fn: Arc<dyn GradFn<f64>>) -> Self {
        Self::F64(AutogradMeta {
            grad: RwLock::new(None),
            grad_fn: Some(grad_fn),
        })
    }

    /// Get f32 autograd metadata if this is F32 variant
    pub fn as_f32(&self) -> Option<&AutogradMeta<f32>> {
        match self {
            Self::F32(meta) => Some(meta),
            _ => None,
        }
    }

    /// Get f64 autograd metadata if this is F64 variant
    pub fn as_f64(&self) -> Option<&AutogradMeta<f64>> {
        match self {
            Self::F64(meta) => Some(meta),
            _ => None,
        }
    }
}

// ============================================================================
// FloatDTypeAutograd - Sealed trait for type-safe autograd storage creation
// ============================================================================

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Extension trait for FloatDType that provides autograd storage creation
///
/// This is a sealed trait - only implemented for f32 and f64.
pub trait FloatDTypeAutograd: FloatDType + sealed::Sealed {
    /// Create AutogradStorage from AutogradMeta<Self>
    fn wrap_autograd(meta: AutogradMeta<Self>) -> AutogradStorage;

    /// Create AutogradStorage with gradient function
    fn wrap_grad_fn(grad_fn: Arc<dyn GradFn<Self>>) -> AutogradStorage;

    /// Create empty AutogradStorage for this type
    fn new_autograd() -> AutogradStorage;
}

impl FloatDTypeAutograd for f32 {
    fn wrap_autograd(meta: AutogradMeta<Self>) -> AutogradStorage {
        AutogradStorage::F32(meta)
    }

    fn wrap_grad_fn(grad_fn: Arc<dyn GradFn<Self>>) -> AutogradStorage {
        AutogradStorage::new_f32_with_grad_fn(grad_fn)
    }

    fn new_autograd() -> AutogradStorage {
        AutogradStorage::new_f32()
    }
}

impl FloatDTypeAutograd for f64 {
    fn wrap_autograd(meta: AutogradMeta<Self>) -> AutogradStorage {
        AutogradStorage::F64(meta)
    }

    fn wrap_grad_fn(grad_fn: Arc<dyn GradFn<Self>>) -> AutogradStorage {
        AutogradStorage::new_f64_with_grad_fn(grad_fn)
    }

    fn new_autograd() -> AutogradStorage {
        AutogradStorage::new_f64()
    }
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
    pub(crate) autograd: Option<AutogradStorage>,
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
/// - **Static Dimensions**: Use `Tensor<f32, Dim<N>>` for compile-time dimension checking
/// - **Dynamic Dimensions**: Use `Tensor<f32, DimDyn>` for runtime-determined dimensions
/// - **Automatic Differentiation**: Track gradients with `.set_requires_grad(true)`
/// - **Eager Fusion**: Operations are fused at call time for optimization
///
/// # Examples
///
/// ```ignore
/// use harp::tensor::{Tensor, Dim2};
///
/// // Static 2D tensor
/// let matrix = Tensor::<f32, Dim2>::zeros([3, 4]);
///
/// // Dynamic dimension tensor
/// let dynamic = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4, 5]);
/// ```
pub struct Tensor<T: TensorDType = f32, D: Dimension = DimDyn> {
    /// Internal tensor data (reference counted for efficient sharing)
    pub(crate) inner: Arc<TensorInner>,
    /// Marker for data type
    pub(crate) _dtype: PhantomData<T>,
    /// Marker for dimension type
    pub(crate) _dim: PhantomData<D>,
}

// Implement Send and Sync for Tensor
unsafe impl<T: TensorDType, D: Dimension> Send for Tensor<T, D> {}
unsafe impl<T: TensorDType, D: Dimension> Sync for Tensor<T, D> {}

impl<T: TensorDType, D: Dimension> Clone for Tensor<T, D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

impl<D: Dimension> Tensor<f32, D> {
    /// Create an explicit branch point in the computation graph (Clone operation)
    ///
    /// Use this when you need to use the same tensor in multiple computation paths.
    /// This creates a Clone operation in the graph that will copy the buffer at execution time.
    ///
    /// Note: Only available for f32 tensors due to TensorRef being f32-typed.
    ///
    /// # Example
    /// ```ignore
    /// let a = x + y;
    /// let a2 = a.fork();       // Clone op added to graph
    /// let b = a.sum();         // a consumed → fusion OK
    /// let c = a2 * 2.0;        // a2 is separate path
    /// ```
    pub fn fork(&self) -> Tensor<f32, D> {
        let input = Arc::new(self.clone().into_dyn());
        let inner = TensorInner::new(
            TensorOp::Clone { input },
            self.inner.view.clone(),
            self.inner.shape.clone(),
            self.inner.dtype.clone(),
        );
        Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

impl<T: TensorDType, D: Dimension> Tensor<T, D> {
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

    /// Convert to a dynamic dimension tensor
    pub fn into_dyn(self) -> Tensor<T, DimDyn> {
        Tensor {
            inner: self.inner,
            _dtype: PhantomData,
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
// Autograd-enabled tensor operations (FloatDTypeAutograd only)
// ============================================================================

impl<T: FloatDTypeAutograd, D: Dimension> Tensor<T, D> {
    /// Enable gradient tracking for this tensor
    ///
    /// Note: This creates a new tensor with gradient tracking enabled.
    /// The original tensor is consumed.
    ///
    /// Only available for types that support autograd (f32, f64).
    pub fn set_requires_grad(self, requires_grad: bool) -> Self {
        if requires_grad && self.inner.autograd.is_none() {
            // Create new inner with autograd
            let inner = TensorInner {
                op: self.inner.op.clone(),
                view: self.inner.view.clone(),
                shape: self.inner.shape.clone(),
                dtype: self.inner.dtype.clone(),
                name: self.inner.name.clone(),
                autograd: Some(T::new_autograd()),
                buffer: RwLock::new(self.inner.buffer.read().unwrap().clone()),
            };
            Tensor {
                inner: Arc::new(inner),
                _dtype: PhantomData,
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
                _dtype: PhantomData,
                _dim: PhantomData,
            }
        } else {
            self
        }
    }
}

// ============================================================================
// DimDyn to static dimension conversion
// ============================================================================

impl<T: TensorDType> Tensor<T, DimDyn> {
    /// Try to convert to a tensor with static dimension.
    ///
    /// Returns `Some(Tensor<T, Dim<N>>)` if the tensor has exactly N dimensions,
    /// otherwise returns `None`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let dyn_tensor: Tensor<f32, DimDyn> = Tensor::ones_dyn(&[2, 3]);
    /// let static_tensor: Tensor<f32, Dim2> = dyn_tensor.try_into_dim().unwrap();
    /// ```
    pub fn try_into_dim<D: Dimension>(&self) -> Option<Tensor<T, D>> {
        if D::NDIM == Some(self.ndim()) {
            Some(Tensor {
                inner: self.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            })
        } else {
            None
        }
    }

    /// Convert to a tensor with static dimension, panicking on mismatch.
    ///
    /// # Panics
    ///
    /// Panics if the tensor's ndim doesn't match the target dimension.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let dyn_tensor: Tensor<f32, DimDyn> = Tensor::ones_dyn(&[2, 3]);
    /// let static_tensor: Tensor<f32, Dim2> = dyn_tensor.into_dimensioned();
    /// ```
    pub fn into_dimensioned<D: Dimension>(&self) -> Tensor<T, D> {
        self.try_into_dim().unwrap_or_else(|| {
            panic!(
                "Cannot convert Tensor with {} dimensions to Dim<{}>",
                self.ndim(),
                D::NDIM.map_or("?".to_string(), |n| n.to_string())
            )
        })
    }

    /// Convert to 0-dimensional tensor (scalar).
    pub fn into_dim0(&self) -> Tensor<T, Dim0> {
        self.into_dimensioned()
    }

    /// Convert to 1-dimensional tensor.
    pub fn into_dim1(&self) -> Tensor<T, Dim1> {
        self.into_dimensioned()
    }

    /// Convert to 2-dimensional tensor.
    pub fn into_dim2(&self) -> Tensor<T, Dim2> {
        self.into_dimensioned()
    }

    /// Convert to 3-dimensional tensor.
    pub fn into_dim3(&self) -> Tensor<T, Dim3> {
        self.into_dimensioned()
    }

    /// Convert to 4-dimensional tensor.
    pub fn into_dim4(&self) -> Tensor<T, Dim4> {
        self.into_dimensioned()
    }

    /// Convert to 5-dimensional tensor.
    pub fn into_dim5(&self) -> Tensor<T, Dim5> {
        self.into_dimensioned()
    }

    /// Convert to 6-dimensional tensor.
    pub fn into_dim6(&self) -> Tensor<T, Dim6> {
        self.into_dimensioned()
    }
}

// ============================================================================
// Backward propagation
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Perform backward propagation from this tensor
    ///
    /// Computes gradients for all tensors in the computation graph that
    /// have `requires_grad = true`.
    ///
    /// Note: Gradient computation is only available for f32 tensors.
    ///
    /// # Panics
    ///
    /// Panics if this tensor does not require gradients.
    pub fn backward(&self) {
        if self.inner.autograd.is_none() {
            panic!("backward() called on tensor that doesn't require gradients");
        }

        // Create initial gradient of ones with same shape
        let initial_grad = Tensor::<f32, DimDyn>::ones_dyn(self.shape());
        self.backward_with(initial_grad);
    }

    /// Perform backward propagation with a custom initial gradient
    pub fn backward_with(&self, grad_output: Tensor<f32, DimDyn>) {
        if let Some(ref autograd_storage) = self.inner.autograd {
            // Get f32 autograd metadata
            let autograd = autograd_storage
                .as_f32()
                .expect("f32 tensor should have f32 autograd storage");

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
    ///
    /// The returned gradient has the same dimension type as the original tensor.
    pub fn grad(&self) -> Option<Tensor<f32, D>> {
        self.inner
            .autograd
            .as_ref()
            .and_then(|storage| storage.as_f32())
            .and_then(|ag| {
                ag.grad.read().unwrap().as_ref().map(|g| {
                    // Convert from DimDyn to D (same underlying data, different type marker)
                    Tensor {
                        inner: g.inner.clone(),
                        _dtype: PhantomData,
                        _dim: PhantomData,
                    }
                })
            })
    }

    /// Reset the gradient to None
    pub fn zero_grad(&self) {
        if let Some(ref autograd_storage) = self.inner.autograd
            && let Some(autograd) = autograd_storage.as_f32()
        {
            *autograd.grad.write().unwrap() = None;
        }
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    pub fn detach(&self) -> Tensor<f32, D> {
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
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros_static() {
        let t = Tensor::<f32, Dim2>::zeros([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 12);
    }

    #[test]
    fn test_tensor_ones_static() {
        let t = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_tensor_full_static() {
        let t = Tensor::<f32, Dim1>::full([10], 2.5);
        assert_eq!(t.shape(), &[10]);
        assert_eq!(t.ndim(), 1);
    }

    #[test]
    fn test_tensor_zeros_dynamic() {
        let t = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4, 5]);
        assert_eq!(t.shape(), &[3, 4, 5]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 60);
    }

    #[test]
    fn test_tensor_input() {
        let t = Tensor::<f32, Dim2>::input("x", [10, 20]);
        assert_eq!(t.shape(), &[10, 20]);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_into_dyn() {
        let static_tensor = Tensor::<f32, Dim2>::zeros([3, 4]);
        let dyn_tensor = static_tensor.into_dyn();
        assert_eq!(dyn_tensor.shape(), &[3, 4]);
    }

    #[test]
    fn test_set_requires_grad() {
        let t = Tensor::<f32, Dim2>::ones([2, 2]).set_requires_grad(true);
        assert!(t.requires_grad());
    }

    #[test]
    fn test_detach() {
        let t = Tensor::<f32, Dim2>::ones([2, 2]).set_requires_grad(true);
        let detached = t.detach();
        assert!(!detached.requires_grad());
    }

    #[test]
    fn test_rand() {
        let t = Tensor::<f32, Dim2>::rand([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_fork() {
        let t = Tensor::<f32, Dim2>::ones([2, 3]);
        let forked = t.fork();

        // fork() creates a new tensor with the same shape
        assert_eq!(forked.shape(), t.shape());
        assert_eq!(forked.dtype(), t.dtype());

        // fork() does not preserve gradient tracking
        let t_grad = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let forked_grad = t_grad.fork();
        assert!(!forked_grad.requires_grad());

        // The forked tensor should have a Clone operation
        assert!(matches!(forked.inner.op, TensorOp::Clone { .. }));
    }

    #[test]
    fn test_into_dimensioned() {
        let dyn_tensor = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4]);
        let static_tensor: Tensor<f32, Dim2> = dyn_tensor.into_dimensioned();
        assert_eq!(static_tensor.shape(), &[3, 4]);
        assert_eq!(static_tensor.ndim(), 2);
    }

    #[test]
    fn test_into_dim1() {
        let dyn_tensor = Tensor::<f32, DimDyn>::ones_dyn(&[5]);
        let static_tensor = dyn_tensor.into_dim1();
        assert_eq!(static_tensor.shape(), &[5]);
    }

    #[test]
    fn test_into_dim2() {
        let dyn_tensor = Tensor::<f32, DimDyn>::ones_dyn(&[3, 4]);
        let static_tensor = dyn_tensor.into_dim2();
        assert_eq!(static_tensor.shape(), &[3, 4]);
    }

    #[test]
    fn test_try_into_dim_success() {
        let dyn_tensor = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let result: Option<Tensor<f32, Dim2>> = dyn_tensor.try_into_dim();
        assert!(result.is_some());
        assert_eq!(result.unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn test_try_into_dim_failure() {
        let dyn_tensor = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3, 4]);
        let result: Option<Tensor<f32, Dim2>> = dyn_tensor.try_into_dim();
        assert!(result.is_none());
    }

    #[test]
    #[should_panic(expected = "Cannot convert Tensor with 3 dimensions to Dim<2>")]
    fn test_into_dimensioned_panic() {
        let dyn_tensor = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3, 4]);
        let _: Tensor<f32, Dim2> = dyn_tensor.into_dimensioned();
    }

    #[test]
    fn test_tensor_type_aliases() {
        // Test that type aliases work correctly
        let t0: Tensor0 = Tensor::<f32, Dim0>::full([], 1.0);
        let t1: Tensor1 = Tensor::<f32, Dim1>::ones([5]);
        let t2: Tensor2 = Tensor::<f32, Dim2>::zeros([3, 4]);
        let t3: Tensor3 = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        let t_dyn: TensorDyn = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);

        assert_eq!(t0.ndim(), 0);
        assert_eq!(t1.ndim(), 1);
        assert_eq!(t2.ndim(), 2);
        assert_eq!(t3.ndim(), 3);
        assert_eq!(t_dyn.ndim(), 2);
    }
}
