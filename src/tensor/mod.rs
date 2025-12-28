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

use log::debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use crate::backend::Buffer;

pub use dimension::{Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
pub use dtype::{NumericDType, TensorDType};
// IntegerDType, SignedIntDType, UnsignedIntDType, NumericInitDType are defined below
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

use crate::ast::{DType, Literal};

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
    /// Zero value for this type
    const ZERO: Self;
    /// One value for this type
    const ONE: Self;

    /// Convert value to Literal for ConstFill operations
    fn to_literal(val: Self) -> Literal;

    /// Create AutogradStorage from AutogradMeta<Self>
    fn wrap_autograd(meta: AutogradMeta<Self>) -> AutogradStorage;

    /// Create AutogradStorage with gradient function
    fn wrap_grad_fn(grad_fn: Arc<dyn GradFn<Self>>) -> AutogradStorage;

    /// Create empty AutogradStorage for this type
    fn new_autograd() -> AutogradStorage;
}

impl FloatDType for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn to_literal(val: Self) -> Literal {
        Literal::F32(val)
    }

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

impl FloatDType for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn to_literal(val: Self) -> Literal {
        Literal::F64(val)
    }

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
pub trait IntegerDType: NumericDType + sealed_int::Sealed {
    /// Zero value for this type
    const ZERO: Self;
    /// One value for this type
    const ONE: Self;

    /// Convert value to Literal for ConstFill operations
    fn to_literal(val: Self) -> Literal;
}

/// Trait for signed integer types (i8, i16, i32, i64)
pub trait SignedIntDType: IntegerDType {}

/// Trait for unsigned integer types (u8, u16, u32, u64)
pub trait UnsignedIntDType: IntegerDType {}

/// Macro to implement IntegerDType for signed integers
macro_rules! impl_signed_int_dtype {
    ($($ty:ty => $literal:ident);+ $(;)?) => {
        $(
            impl IntegerDType for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
                fn to_literal(val: Self) -> Literal {
                    Literal::$literal(val)
                }
            }
            impl SignedIntDType for $ty {}
        )+
    };
}

/// Macro to implement IntegerDType for unsigned integers
macro_rules! impl_unsigned_int_dtype {
    ($($ty:ty => $literal:ident);+ $(;)?) => {
        $(
            impl IntegerDType for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
                fn to_literal(val: Self) -> Literal {
                    Literal::$literal(val)
                }
            }
            impl UnsignedIntDType for $ty {}
        )+
    };
}

impl_signed_int_dtype!(
    i8  => I8;
    i16 => I16;
    i32 => I32;
    i64 => I64;
);

impl_unsigned_int_dtype!(
    u8  => U8;
    u16 => U16;
    u32 => U32;
    u64 => U64;
);

// ============================================================================
// NumericInitDType - Common trait for numeric types with initialization
// ============================================================================

/// Trait for numeric types that support tensor initialization
///
/// This is a common trait that both FloatDType and IntegerDType implement,
/// allowing unified initialization methods (zeros, ones, full, input) for
/// all numeric tensor types.
pub trait NumericInitDType: NumericDType {
    /// Zero value for this type
    const ZERO: Self;
    /// One value for this type
    const ONE: Self;

    /// Convert value to Literal for ConstFill operations
    fn to_literal(val: Self) -> Literal;
}

// Blanket implementations for FloatDType and IntegerDType
impl<T: FloatDType> NumericInitDType for T {
    const ZERO: Self = <T as FloatDType>::ZERO;
    const ONE: Self = <T as FloatDType>::ONE;

    fn to_literal(val: Self) -> Literal {
        <T as FloatDType>::to_literal(val)
    }
}

// Note: IntegerDType implementation is done via macro below

/// Macro to implement NumericInitDType for integer types
macro_rules! impl_numeric_init_for_int {
    ($($ty:ty => $literal:ident);+ $(;)?) => {
        $(
            impl NumericInitDType for $ty {
                const ZERO: Self = 0;
                const ONE: Self = 1;
                fn to_literal(val: Self) -> Literal {
                    Literal::$literal(val)
                }
            }
        )+
    };
}

impl_numeric_init_for_int!(
    i8  => I8;
    i16 => I16;
    i32 => I32;
    i64 => I64;
    u8  => U8;
    u16 => U16;
    u32 => U32;
    u64 => U64;
);

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

    /// Clone the buffer option using Buffer::clone_buffer()
    pub(crate) fn clone_buffer(&self) -> Option<Box<dyn Buffer>> {
        self.buffer
            .read()
            .unwrap()
            .as_ref()
            .map(|buf| buf.clone_buffer())
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
        let input = self.as_input_ref();
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
// Autograd-enabled tensor operations (FloatDType only)
// ============================================================================

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
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
                buffer: RwLock::new(self.inner.clone_buffer()),
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
                buffer: RwLock::new(self.inner.clone_buffer()),
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
        if self.inner.autograd.is_none() {
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
        if let Some(ref autograd_storage) = self.inner.autograd {
            // Get f32 autograd metadata
            let autograd = autograd_storage
                .as_f32()
                .expect("f32 tensor should have f32 autograd storage");

            // Accumulate gradient
            {
                let mut grad = autograd.grad.write().unwrap();
                if let Some(existing) = grad.take() {
                    // Add to existing gradient (this addition is also tracked in the graph)
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

                // Propagate gradients to each input tensor (using create_graph version)
                for (input, grad) in inputs.into_iter().zip(input_grads.into_iter()) {
                    if input.requires_grad() {
                        input.backward_with_create_graph(grad);
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
            buffer: RwLock::new(self.inner.clone_buffer()),
        };
        Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// f64 Autograd Methods
// ============================================================================

impl<D: Dimension> Tensor<f64, D> {
    /// Perform backward propagation starting from this tensor
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
        let initial_grad = Tensor::<f64, DimDyn>::ones_dyn(self.shape());
        self.backward_with(initial_grad);
    }

    /// Perform backward propagation with a custom initial gradient
    pub fn backward_with(&self, grad_output: Tensor<f64, DimDyn>) {
        if let Some(ref autograd_storage) = self.inner.autograd {
            // Get f64 autograd metadata
            let autograd = autograd_storage
                .as_f64()
                .expect("f64 tensor should have f64 autograd storage");

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
        if self.inner.autograd.is_none() {
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
        if let Some(ref autograd_storage) = self.inner.autograd {
            // Get f64 autograd metadata
            let autograd = autograd_storage
                .as_f64()
                .expect("f64 tensor should have f64 autograd storage");

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

                // Propagate gradients to each input tensor (using create_graph version)
                for (input, grad) in inputs.into_iter().zip(input_grads.into_iter()) {
                    if input.requires_grad() {
                        input.backward_with_create_graph(grad);
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
    pub fn grad(&self) -> Option<Tensor<f64, D>> {
        self.inner
            .autograd
            .as_ref()
            .and_then(|storage| storage.as_f64())
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
            && let Some(autograd) = autograd_storage.as_f64()
        {
            *autograd.grad.write().unwrap() = None;
        }
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    pub fn detach(&self) -> Tensor<f64, D> {
        let inner = TensorInner {
            op: self.inner.op.clone(),
            view: self.inner.view.clone(),
            shape: self.inner.shape.clone(),
            dtype: self.inner.dtype.clone(),
            name: self.inner.name.clone(),
            autograd: None,
            buffer: RwLock::new(self.inner.clone_buffer()),
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
}
