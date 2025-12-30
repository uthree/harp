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
//! Operations are fused at call time using the unified MapReduce variant.
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
pub mod stringify;

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
use log::debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use crate::backend::Buffer;

pub use dimension::{Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, Dim7, Dim8, DimDyn, Dimension};
pub use dtype::{NumericDType, TensorDType};
// IntegerDType, SignedIntDType, UnsignedIntDType are defined below
pub use forward::ForwardError;
pub use ops::{ElementwiseOp, InputRef, ReduceOp, TensorOp};
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
// GradFn - New statically-typed gradient function trait
// ============================================================================

/// Gradient function trait with static dimension typing (new version)
///
/// This trait is implemented by operations that can compute gradients.
/// Generic over T (floating-point type) and D (output gradient dimension).
/// Each implementation holds its input tensors and propagates gradients internally.
///
/// Unlike `GradFn<T>`, this trait:
/// - Uses static dimension typing (D) instead of DimDyn
/// - Does not return gradients; implementations call `backward_with` on inputs directly
/// - Does not have `inputs()` method; each struct holds its own inputs
pub trait GradFn<T: FloatDType, D: Dimension>: Send + Sync {
    /// Compute and propagate gradients to inputs
    ///
    /// # Arguments
    /// * `grad_output` - Gradient flowing back from the output (dimension D)
    ///
    /// The implementation is responsible for calling `backward_with` on its inputs.
    fn backward(&self, grad_output: &Tensor<T, D>);

    /// Get the name of this gradient function (for debugging)
    fn name(&self) -> &'static str;
}

// ============================================================================
// AutogradMeta - New statically-typed autograd metadata
// ============================================================================

/// Autograd metadata with static dimension typing (new version)
///
/// The presence of this struct indicates that gradient tracking is enabled.
/// Unlike `AutogradMeta<T>`, this version preserves the dimension type D.
pub struct AutogradMeta<T: FloatDType, D: Dimension> {
    /// Stored gradient with static dimension (populated after backward())
    pub(crate) grad: RwLock<Option<Arc<Tensor<T, D>>>>,
    /// Gradient function for backpropagation (uses GradFn)
    pub(crate) grad_fn: Option<Arc<dyn GradFn<T, D>>>,
}

impl<T: FloatDType, D: Dimension> AutogradMeta<T, D> {
    /// Create new autograd metadata without gradient function (leaf tensor)
    pub fn new() -> Self {
        Self {
            grad: RwLock::new(None),
            grad_fn: None,
        }
    }

    /// Create new autograd metadata with gradient function
    pub fn with_grad_fn(grad_fn: Arc<dyn GradFn<T, D>>) -> Self {
        Self {
            grad: RwLock::new(None),
            grad_fn: Some(grad_fn),
        }
    }
}

impl<T: FloatDType, D: Dimension> Default for AutogradMeta<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// FloatDType - Sealed trait for floating-point types with autograd support
// ============================================================================

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Trait for floating-point types with autograd support (f32, f64, future f16/bf16)
///
/// This is a sealed trait - only implemented for f32 and f64 (and future half-precision types).
/// Provides transcendental functions (sin, cos, exp, log, etc.), gradient computation, and
/// floor/ceil/round operations.
pub trait FloatDType: NumericDType + sealed::Sealed {
    /// Negative infinity value for this type
    const NEG_INF: Self;

    /// Mathematical constant: log₂(e)
    const LOG2_E: Self;

    /// Mathematical constant: ln(2)
    const LN_2: Self;

    /// Mathematical constant: π/2
    const FRAC_PI_2: Self;

    /// Zero value
    const ZERO: Self;

    /// One value
    const ONE: Self;

    /// Two value
    const TWO: Self;

    /// Convert from usize to this type
    fn from_usize(val: usize) -> Self;

    /// Square root of a scalar value
    fn sqrt(self) -> Self;

    /// Small epsilon for numerical stability
    const EPSILON: Self;
}

impl FloatDType for f32 {
    const NEG_INF: Self = f32::NEG_INFINITY;
    const LOG2_E: Self = std::f32::consts::LOG2_E;
    const LN_2: Self = std::f32::consts::LN_2;
    const FRAC_PI_2: Self = std::f32::consts::FRAC_PI_2;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const EPSILON: Self = 1e-8;

    fn from_usize(val: usize) -> Self {
        val as f32
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl FloatDType for f64 {
    const NEG_INF: Self = f64::NEG_INFINITY;
    const LOG2_E: Self = std::f64::consts::LOG2_E;
    const LN_2: Self = std::f64::consts::LN_2;
    const FRAC_PI_2: Self = std::f64::consts::FRAC_PI_2;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const EPSILON: Self = 1e-8;

    fn from_usize(val: usize) -> Self {
        val as f64
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

// ============================================================================
// IntegerDType - Sealed trait for integer types
// ============================================================================

mod sealed_int {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Trait for integer types (i8, i16, i32, i64, u8, u16, u32, u64)
///
/// This is a sealed trait. Provides:
/// - Bitwise operations (and, or, xor, shl, shr)
/// - Integer division and remainder
pub trait IntegerDType: NumericDType + sealed_int::Sealed {}

/// Trait for signed integer types (i8, i16, i32, i64)
pub trait SignedIntDType: IntegerDType {}

/// Trait for unsigned integer types (u8, u16, u32, u64)
pub trait UnsignedIntDType: IntegerDType {}

/// Macro to implement IntegerDType for signed integers
macro_rules! impl_signed_int_dtype {
    ($($ty:ty);+ $(;)?) => {
        $(
            impl IntegerDType for $ty {}
            impl SignedIntDType for $ty {}
        )+
    };
}

/// Macro to implement IntegerDType for unsigned integers
macro_rules! impl_unsigned_int_dtype {
    ($($ty:ty);+ $(;)?) => {
        $(
            impl IntegerDType for $ty {}
            impl UnsignedIntDType for $ty {}
        )+
    };
}

impl_signed_int_dtype!(i8; i16; i32; i64);

impl_unsigned_int_dtype!(u8; u16; u32; u64);

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
    /// Executed buffer data (populated after realize())
    pub(crate) buffer: RwLock<Option<Box<dyn Buffer>>>,
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
            buffer: RwLock::new(None),
        }
    }

    /// 演算の種類を取得
    pub fn op(&self) -> &TensorOp {
        &self.op
    }

    /// Viewを取得
    pub fn view(&self) -> &View {
        &self.view
    }

    /// 形状を取得
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// データ型を取得
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// 名前を取得
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// バッファへのアクセス（RwLock経由）
    pub fn buffer(&self) -> &RwLock<Option<Box<dyn Buffer>>> {
        &self.buffer
    }

    /// バッファが存在するか
    pub fn has_buffer(&self) -> bool {
        self.buffer.read().unwrap().is_some()
    }

    /// この演算が融合可能かどうかを判定
    ///
    /// 融合可能な演算は、lowererによって親カーネルに式として展開される。
    /// 融合不可能な演算は、realize_recursiveで先にバッファを作成する必要がある。
    pub fn is_fusable(&self) -> bool {
        match &self.op {
            // Elementwise MapReduce（reduce_op: None, axes: []）は融合可能
            TensorOp::MapReduce {
                reduce_op: None,
                axes,
                ..
            } if axes.is_empty() => true,
            // 定数は融合可能
            TensorOp::Const(_) | TensorOp::ConstFill(_) => true,
            // View操作も融合可能（形状変換のみでバッファ不要）
            TensorOp::View { .. } => true,
            // Buffer/Executed/Randは入力ソースなので融合可能とみなす
            TensorOp::Buffer { .. } | TensorOp::Executed | TensorOp::Rand => true,
            // その他は融合バリア
            _ => false,
        }
    }

    /// 入力のrealizeをスキップできるかどうかを判定
    ///
    /// 以下の条件を満たす場合、入力は融合可能でありrealizeをスキップできる:
    /// 1. Const/ConstFill: 常にインライン化される（バッファ不要）
    /// 2. その他: `is_fusable()` && `Arc::strong_count == 1`
    ///
    /// strong_count > 1の場合、他の演算も依存しているためバッファを作成する必要がある。
    /// ただし Const/ConstFill は常にインライン化されるため例外。
    #[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
    fn can_skip_realize(input: &InputRef) -> bool {
        // Const/ConstFill は常にスキップ（lowererでインライン化される）
        if matches!(input.op, TensorOp::Const(_) | TensorOp::ConstFill(_)) {
            return true;
        }
        // その他は strong_count == 1 かつ is_fusable() の場合のみスキップ
        Arc::strong_count(input) == 1 && input.is_fusable()
    }

    /// バッファデータをホストに読み出し
    pub fn read_buffer(&self) -> Option<Vec<u8>> {
        self.buffer
            .read()
            .ok()?
            .as_ref()
            .and_then(|b| b.read_to_host().ok())
    }

    /// 自身を再帰的にrealizeする
    ///
    /// 1. 既にバッファがあればスキップ
    /// 2. MapReduce操作なら入力を先にrealize（融合可能な入力はスキップ）
    /// 3. 自身をrealize_core()でrealize
    ///
    /// 融合ロジック:
    /// - `can_skip_realize()` が true の入力はrealizeをスキップし、lowererで融合
    /// - strong_count > 1 の入力は他の演算も依存しているためrealizeする
    ///
    /// View/Buffer/Executed等はrealizeしない（親のMapReduceが直接参照する）
    #[cfg(any(all(feature = "metal", target_os = "macos"), feature = "opencl"))]
    pub fn realize_recursive(&self) -> Result<(), String> {
        // 既にバッファがあればスキップ
        if self.buffer.read().unwrap().is_some() {
            return Ok(());
        }

        // 非計算ノード・融合可能ノードの処理
        match &self.op {
            // View: 自身はバッファを持たない
            // Viewがrealizeのルートとして呼ばれた場合、入力の融合判定に基づきrealize
            TensorOp::View { input } => {
                if !Self::can_skip_realize(input) {
                    input.realize_recursive()?;
                }
                return Ok(());
            }
            // ソースノード: 処理不要
            TensorOp::Buffer { .. } | TensorOp::Executed => return Ok(()),
            // Const(0.0)はlowererが直接処理するのでスキップ
            TensorOp::Const(_) => return Ok(()),
            // Note: ConstFillはバッファを生成するのでスキップしない
            _ => {}
        }

        // 融合バリア（Contiguous/Clone）: 入力を必ずrealize
        match &self.op {
            TensorOp::Contiguous { input } | TensorOp::Clone { input } => {
                input.realize_recursive()?;
            }
            _ => {}
        }

        // MapReduce: 融合可能な入力はスキップ
        if let TensorOp::MapReduce { inputs, .. } = &self.op {
            for input in inputs {
                // can_skip_realize(): is_fusable() && strong_count == 1
                if Self::can_skip_realize(input) {
                    debug!(
                        "Fusion: FUSED input (strong_count={}, op={})",
                        Arc::strong_count(input),
                        input.op.name()
                    );
                } else {
                    debug!(
                        "Fusion: realize input (strong_count={}, is_fusable={}, op={})",
                        Arc::strong_count(input),
                        input.is_fusable(),
                        input.op.name()
                    );
                    input.realize_recursive()?;
                }
            }
        }

        // Concat: 全入力がバリア（必ずrealize）
        if let TensorOp::Concat { inputs, .. } = &self.op {
            for input in inputs {
                input.realize_recursive()?;
            }
        }

        // 自身をrealize
        self.realize_core().map_err(|e| e.to_string())
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
    /// Type-safe autograd metadata (only used when T: FloatDType)
    /// This is the new typed autograd system - will replace TensorInner.autograd
    pub(crate) autograd_meta: Option<Arc<dyn std::any::Any + Send + Sync>>,
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
            autograd_meta: self.autograd_meta.clone(),
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
        let input = self.as_input_ref();
        let inner = TensorInner::new(
            TensorOp::Clone { input },
            self.inner.view.clone(),
            self.inner.shape.clone(),
            self.inner.dtype.clone(),
        );
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

impl<T: TensorDType, D: Dimension> Tensor<T, D> {
    /// Convert to InputRef for use in TensorOp
    ///
    /// This creates a type-erased reference that can be stored in TensorOp
    /// for graph traversal while maintaining the computation graph.
    pub fn as_input_ref(&self) -> InputRef {
        Arc::clone(&self.inner) as InputRef
    }

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
    ///
    /// Returns true if gradient tracking is enabled for this tensor.
    pub fn requires_grad(&self) -> bool {
        self.autograd_meta.is_some()
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

    /// 型変換（Cast）
    ///
    /// テンソルを別の型に変換する。MapReduceとして実装されているため、
    /// 他のelementwise演算と融合可能。
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a: Tensor<f32, Dim2> = Tensor::ones([2, 3]);
    /// let b: Tensor<i32, Dim2> = a.cast();
    /// ```
    pub fn cast<U: TensorDType>(&self) -> Tensor<U, D> {
        let view = self.inner.view.clone();
        let shape = self.inner.shape.clone();
        let op = TensorOp::cast(self.as_input_ref(), U::DTYPE);
        let inner = TensorInner::new(op, view, shape, U::DTYPE);
        Tensor {
            inner: Arc::new(inner),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Autograd-enabled tensor operations (FloatDType only)
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Convert to a dynamic dimension tensor, preserving gradient tracking
    ///
    /// If this tensor requires gradients, the result will also require gradients
    /// and include a backward function that propagates gradients back.
    pub fn into_dyn(self) -> Tensor<T, DimDyn> {
        if self.requires_grad() {
            let result = Tensor {
                inner: self.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_fn = primops::IntoDynBackward::<T, D>::new(self);
            result.with_grad_fn(Arc::new(grad_fn))
        } else {
            Tensor {
                inner: self.inner,
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            }
        }
    }

    /// Enable gradient tracking for this tensor
    ///
    /// Note: This creates a new tensor with gradient tracking enabled.
    /// The original tensor is consumed.
    ///
    /// Only available for types that support autograd (f32, f64).
    pub fn set_requires_grad(self, requires_grad: bool) -> Self {
        if requires_grad && self.autograd_meta.is_none() {
            // Enable gradient tracking
            Tensor {
                inner: self.inner,
                autograd_meta: Some(Arc::new(AutogradMeta::<T, D>::new())),
                _dtype: PhantomData,
                _dim: PhantomData,
            }
        } else if !requires_grad && self.autograd_meta.is_some() {
            // Disable gradient tracking
            Tensor {
                inner: self.inner,
                autograd_meta: None,
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
    /// Note: The autograd_meta is preserved but may contain a different dimension type.
    /// The backward_with method handles this mismatch by falling back to DimDyn.
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
                autograd_meta: self.autograd_meta.clone(),
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

    /// Convert to 7-dimensional tensor.
    pub fn into_dim7(&self) -> Tensor<T, Dim7> {
        self.into_dimensioned()
    }

    /// Convert to 8-dimensional tensor.
    pub fn into_dim8(&self) -> Tensor<T, Dim8> {
        self.into_dimensioned()
    }
}

// ============================================================================
// Backward propagation
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Perform backward propagation while retaining the computation graph
    ///
    /// This enables higher-order derivatives (e.g., second derivatives) by
    /// creating gradients that themselves have `requires_grad = true`.
    ///
    /// # Returns
    ///
    /// The initial gradient tensor, which can be used to compute higher-order
    /// derivatives by calling `.grad()` on it after backward propagation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Compute second derivative: d²(x³)/dx² = 6x
    /// let x = Tensor::<f32, Dim1>::full([1], 2.0).set_requires_grad(true);
    /// let y = &x * &x * &x;  // y = x³
    /// let grad_y = y.backward_create_graph();
    /// // First derivative: dy/dx = 3x² (stored in x.grad())
    /// // grad_y has requires_grad=true, so we can differentiate again
    /// grad_y.backward();
    /// // Second derivative: d²y/dx² = 6x (now in x.grad())
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if this tensor does not require gradients.
    pub fn backward_create_graph(&self) -> Tensor<f32, DimDyn> {
        if self.autograd_meta.is_none() {
            panic!("backward_create_graph() called on tensor that doesn't require gradients");
        }

        // Create initial gradient with requires_grad=true to build computation graph
        let initial_grad = Tensor::<f32, DimDyn>::ones_dyn(self.shape()).set_requires_grad(true);
        self.backward_with_create_graph(initial_grad.clone());
        initial_grad
    }

    /// Perform backward propagation with a custom initial gradient while retaining
    /// the computation graph for higher-order derivatives.
    ///
    /// Unlike `backward_with`, gradients computed during this backward pass will
    /// themselves have `requires_grad = true`, allowing further differentiation.
    pub fn backward_with_create_graph(&self, grad_output: Tensor<f32, DimDyn>) {
        let typed_grad: Tensor<f32, D> = Tensor {
            inner: grad_output.inner.clone(),
            autograd_meta: None,
            _dtype: std::marker::PhantomData,
            _dim: std::marker::PhantomData,
        };
        self.backward_with(typed_grad);
    }

    /// Get the accumulated gradient for this tensor
    ///
    /// Returns None if backward() hasn't been called or if this tensor
    /// doesn't require gradients.
    ///
    /// The returned gradient has the same dimension type as the original tensor.
    pub fn grad(&self) -> Option<Tensor<f32, D>> {
        self.grad_typed()
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    pub fn detach(&self) -> Tensor<f32, D> {
        Tensor {
            inner: self.inner.clone(),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// f64 Autograd Methods
// ============================================================================

impl<D: Dimension> Tensor<f64, D> {
    /// Perform backward propagation while retaining the computation graph
    ///
    /// This enables higher-order derivatives (e.g., second derivatives) by
    /// creating gradients that themselves have `requires_grad = true`.
    ///
    /// # Returns
    ///
    /// The initial gradient tensor, which can be used to compute higher-order
    /// derivatives by calling `.grad()` on it after backward propagation.
    ///
    /// # Panics
    ///
    /// Panics if this tensor does not require gradients.
    pub fn backward_create_graph(&self) -> Tensor<f64, DimDyn> {
        if self.autograd_meta.is_none() {
            panic!("backward_create_graph() called on tensor that doesn't require gradients");
        }

        // Create initial gradient with requires_grad=true to build computation graph
        let initial_grad = Tensor::<f64, DimDyn>::ones_dyn(self.shape()).set_requires_grad(true);
        self.backward_with_create_graph(initial_grad.clone());
        initial_grad
    }

    /// Perform backward propagation with a custom initial gradient while retaining
    /// the computation graph for higher-order derivatives.
    pub fn backward_with_create_graph(&self, grad_output: Tensor<f64, DimDyn>) {
        let typed_grad: Tensor<f64, D> = Tensor {
            inner: grad_output.inner.clone(),
            autograd_meta: None,
            _dtype: std::marker::PhantomData,
            _dim: std::marker::PhantomData,
        };
        self.backward_with(typed_grad);
    }

    /// Get the accumulated gradient for this tensor
    ///
    /// Returns None if backward() hasn't been called or if this tensor
    /// doesn't require gradients.
    ///
    /// The returned gradient has the same dimension type as the original tensor.
    pub fn grad(&self) -> Option<Tensor<f64, D>> {
        self.grad_typed()
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    pub fn detach(&self) -> Tensor<f64, D> {
        Tensor {
            inner: self.inner.clone(),
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Generic autograd methods for FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Reset the gradient to None
    pub fn zero_grad(&self) {
        if let Some(ref autograd_arc) = self.autograd_meta
            && let Some(autograd) = autograd_arc.downcast_ref::<AutogradMeta<T, D>>()
        {
            *autograd.grad.write().unwrap() = None;
        }
    }

    /// Get the accumulated gradient for this tensor (generic version)
    ///
    /// Returns None if backward() hasn't been called or if this tensor
    /// doesn't require gradients.
    ///
    /// The returned gradient has the same dimension type as the original tensor.
    pub fn grad_generic(&self) -> Option<Tensor<T, D>> {
        self.grad_typed()
    }

    // ========================================================================
    // Autograd Methods
    // ========================================================================

    /// Get typed autograd metadata
    fn autograd_meta_typed(&self) -> Option<&AutogradMeta<T, D>> {
        self.autograd_meta
            .as_ref()
            .and_then(|arc| arc.downcast_ref::<AutogradMeta<T, D>>())
    }

    /// Perform backward propagation (typed version)
    ///
    /// Creates an initial gradient of ones and propagates backwards.
    /// Uses the new typed autograd system with static dimension tracking.
    pub fn backward(&self) {
        if self.autograd_meta.is_none() {
            panic!("backward() called on tensor that doesn't require gradients");
        }
        // Create ones tensor with dynamic shape, then cast dimension type
        let ones_dyn = Tensor::<T, DimDyn>::ones_dyn(self.shape());
        let initial_grad = Tensor::<T, D> {
            inner: ones_dyn.inner,
            autograd_meta: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        };
        self.backward_with(initial_grad);
    }

    /// Perform backward propagation with a custom initial gradient (typed version)
    ///
    /// This is the core of the new typed autograd system. The gradient function
    /// handles propagation to inputs internally, maintaining static dimension typing.
    ///
    /// If the tensor's autograd_meta has a different dimension type (e.g., DimDyn),
    /// this method will convert the gradient and use that instead.
    pub fn backward_with(&self, grad_output: Tensor<T, D>) {
        if let Some(ref autograd_arc) = self.autograd_meta {
            // Try to downcast to AutogradMeta<T, D>
            if let Some(autograd) = autograd_arc.downcast_ref::<AutogradMeta<T, D>>() {
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

                // grad_fn handles propagation to inputs (no return value)
                if let Some(ref grad_fn) = autograd.grad_fn {
                    grad_fn.backward(&grad_output);
                }
            } else if let Some(autograd) = autograd_arc.downcast_ref::<AutogradMeta<T, DimDyn>>() {
                // Fallback: convert grad_output to DimDyn and use that
                let grad_dyn: Tensor<T, DimDyn> = Tensor {
                    inner: grad_output.inner.clone(),
                    autograd_meta: None,
                    _dtype: PhantomData,
                    _dim: PhantomData,
                };

                // Accumulate gradient
                {
                    let mut grad = autograd.grad.write().unwrap();
                    if let Some(existing) = grad.take() {
                        let new_grad = &(*existing) + &grad_dyn;
                        *grad = Some(Arc::new(new_grad));
                    } else {
                        *grad = Some(Arc::new(grad_dyn.clone()));
                    }
                }

                // grad_fn handles propagation to inputs
                if let Some(ref grad_fn) = autograd.grad_fn {
                    grad_fn.backward(&grad_dyn);
                }
            }
        }
    }

    /// Get the typed gradient for this tensor
    ///
    /// Returns the gradient with static dimension type D.
    /// If the tensor's autograd_meta has a different dimension type (e.g., DimDyn),
    /// this method will try to convert the gradient to the requested type.
    pub fn grad_typed(&self) -> Option<Tensor<T, D>> {
        // First try the exact dimension match
        if let Some(ag) = self.autograd_meta_typed() {
            return ag.grad.read().unwrap().as_ref().map(|g| (**g).clone());
        }

        // Fallback: try DimDyn and convert
        if let Some(ref autograd_arc) = self.autograd_meta
            && let Some(autograd) = autograd_arc.downcast_ref::<AutogradMeta<T, DimDyn>>()
        {
            return autograd.grad.read().unwrap().as_ref().map(|g| Tensor {
                inner: g.inner.clone(),
                autograd_meta: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            });
        }

        None
    }

    /// Create a new tensor with typed autograd and a grad_fn
    pub(crate) fn with_grad_fn(mut self, grad_fn: Arc<dyn GradFn<T, D>>) -> Self {
        self.autograd_meta = Some(Arc::new(AutogradMeta::<T, D>::with_grad_fn(grad_fn)));
        self
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

    // =========================================================================
    // realize_recursive tests
    // =========================================================================

    #[test]
    fn test_realize_recursive_handles_contiguous_input() {
        // Test that Contiguous { input } has its input realized first
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = &a + &a; // MapReduce operation
        let c = b.contiguous(); // Contiguous wrapping a compute node

        // Verify the operation structure
        assert!(matches!(c.inner.op, TensorOp::Contiguous { .. }));

        // The input to Contiguous should be the MapReduce
        if let TensorOp::Contiguous { input } = &c.inner.op {
            assert!(matches!(input.op, TensorOp::MapReduce { .. }));
        }
    }

    #[test]
    fn test_realize_recursive_handles_clone_input() {
        // Test that Clone { input } has its input realized first
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = &a + &a; // MapReduce operation
        let c = b.fork(); // Clone wrapping a compute node

        // Verify the operation structure
        assert!(matches!(c.inner.op, TensorOp::Clone { .. }));

        // The input to Clone should be the MapReduce
        if let TensorOp::Clone { input } = &c.inner.op {
            assert!(matches!(input.op, TensorOp::MapReduce { .. }));
        }
    }

    #[test]
    fn test_realize_recursive_handles_concat_inputs() {
        // Test that Concat { inputs } has all inputs realized first
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = &a + &a; // MapReduce operation
        let c = &a * &a; // Another MapReduce operation
        let d = Tensor::concat(&[&b, &c], 0); // Concat with compute inputs

        // Verify the operation structure
        assert!(matches!(d.inner.op, TensorOp::Concat { .. }));

        // The inputs to Concat should be MapReduce operations
        if let TensorOp::Concat { inputs, .. } = &d.inner.op {
            assert_eq!(inputs.len(), 2);
            assert!(matches!(inputs[0].op, TensorOp::MapReduce { .. }));
            assert!(matches!(inputs[1].op, TensorOp::MapReduce { .. }));
        }
    }

    // =========================================================================
    // Fusion tests
    // =========================================================================

    #[test]
    fn test_is_fusable_elementwise_mapreduce() {
        // Elementwise MapReduce (Add) should be fusable
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = &a + &b;
        assert!(
            c.inner.is_fusable(),
            "Elementwise MapReduce should be fusable"
        );
    }

    #[test]
    fn test_is_fusable_reduce_not_fusable() {
        // Reduce operations should NOT be fusable
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.sum(0);
        assert!(
            !b.inner.is_fusable(),
            "Reduce operations should not be fusable"
        );
    }

    #[test]
    fn test_is_fusable_const() {
        // Const should be fusable
        let a = Tensor::<f32, Dim2>::zeros([2, 3]);
        assert!(a.inner.is_fusable(), "Const should be fusable");
    }

    #[test]
    fn test_is_fusable_const_fill() {
        // ConstFill should be fusable
        let a = Tensor::<f32, Dim2>::full([2, 3], 1.0);
        assert!(a.inner.is_fusable(), "ConstFill should be fusable");
    }

    #[test]
    fn test_is_fusable_view() {
        // View should be fusable
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.reshape([6]);
        assert!(b.inner.is_fusable(), "View should be fusable");
    }

    #[test]
    fn test_is_fusable_contiguous_not_fusable() {
        // Contiguous should NOT be fusable (it's a fusion barrier)
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.contiguous();
        assert!(
            !b.inner.is_fusable(),
            "Contiguous should not be fusable (fusion barrier)"
        );
    }

    #[test]
    fn test_is_fusable_clone_not_fusable() {
        // Clone should NOT be fusable (it's a fusion barrier)
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = a.fork();
        assert!(
            !b.inner.is_fusable(),
            "Clone should not be fusable (fusion barrier)"
        );
    }

    #[test]
    fn test_fusion_chain_no_intermediate_buffer() {
        // When chaining elementwise operations, intermediate results should NOT have buffers
        // until the final result is realized
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = &a + &b; // Intermediate
        let d = &c * 2.0f32; // Final

        // Before realize: neither c nor d should have buffers
        assert!(
            !c.inner.has_buffer(),
            "Intermediate c should not have buffer before realize"
        );
        assert!(
            !d.inner.has_buffer(),
            "Final d should not have buffer before realize"
        );

        // c is fusable, so it should remain without buffer when d is asked to be realized
        // (in terms of structure, not actual execution)
        assert!(c.inner.is_fusable(), "c should be fusable");
        assert!(d.inner.is_fusable(), "d should be fusable");
    }

    #[test]
    fn test_fusion_barrier_forces_realize() {
        // When a fusion barrier is encountered, inputs should be realized
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = &a + &b; // Fusable intermediate
        let d = c.sum(0); // Reduce = fusion barrier

        // c is fusable
        assert!(c.inner.is_fusable(), "c should be fusable");
        // d is NOT fusable (reduce)
        assert!(!d.inner.is_fusable(), "d (reduce) should not be fusable");
    }

    #[test]
    fn test_view_allows_fusion_through() {
        // View operations should allow fusion to continue through them
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = &a + &b; // Fusable
        let d = c.reshape([6]); // View - should also be fusable
        let e = &d * 2.0f32; // Should still be fusable

        assert!(c.inner.is_fusable(), "c should be fusable");
        assert!(d.inner.is_fusable(), "d (view) should be fusable");
        assert!(e.inner.is_fusable(), "e should be fusable after view");
    }

    // =========================================================================
    // Cast tests
    // =========================================================================

    #[test]
    fn test_cast_f32_to_i32() {
        // Cast from f32 to i32
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b: Tensor<i32, Dim2> = a.cast();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(*b.dtype(), DType::I32);
    }

    #[test]
    fn test_cast_i32_to_f32() {
        // Cast from i32 to f32
        let a = Tensor::<i32, Dim2>::full([2, 3], 5);
        let b: Tensor<f32, Dim2> = a.cast();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(*b.dtype(), DType::F32);
    }

    #[test]
    fn test_cast_is_fusable() {
        // Cast should be fusable (it's an elementwise MapReduce)
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b: Tensor<i32, Dim2> = a.cast();
        assert!(b.inner.is_fusable(), "Cast should be fusable");
    }

    #[test]
    fn test_cast_fusion_with_add() {
        // Cast can be fused with other elementwise operations
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = &a + &a; // Elementwise add
        let c: Tensor<i32, Dim2> = b.cast(); // Cast (also elementwise)

        // Both should be fusable
        assert!(b.inner.is_fusable(), "Add should be fusable");
        assert!(c.inner.is_fusable(), "Cast should be fusable");

        // Verify the cast operation is represented as MapReduce
        assert!(
            matches!(c.inner.op, TensorOp::MapReduce { .. }),
            "Cast should be MapReduce"
        );
    }

    #[test]
    fn test_cast_preserves_shape() {
        // Cast should preserve shape
        let a = Tensor::<f32, Dim3>::ones([2, 3, 4]);
        let b: Tensor<i64, Dim3> = a.cast();
        assert_eq!(b.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_cast_f64_to_f32() {
        // Cast from f64 to f32
        let a = Tensor::<f64, Dim1>::full([10], 2.5);
        let b: Tensor<f32, Dim1> = a.cast();
        assert_eq!(b.shape(), &[10]);
        assert_eq!(*b.dtype(), DType::F32);
    }
}
