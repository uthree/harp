//! Binary primitive operations
//!
//! These operations support NumericDType (f32, f64, integers).
//! Gradient tracking is available for FloatDType tensors (f32, f64).
//!
//! - Add: element-wise addition
//! - Mul: element-wise multiplication
//! - Max: element-wise maximum

use std::marker::PhantomData;
use std::ops::Add;
use std::ops::Mul;
use std::sync::Arc;

use crate::ast::{DType, Literal};
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    DimDyn, Dimension, ElementwiseOp, FloatDType, GradFn, GradFnTyped, IntegerDType, NumericDType,
    Tensor, TensorInner, TensorOp,
};

use super::grad::reduce_grad_for_broadcast_generic;

// ============================================================================
// Binary Gradients (generic over FloatDType)
// ============================================================================

/// Gradient for Add: z = a + b
/// ∂L/∂a = ∂L/∂z, ∂L/∂b = ∂L/∂z
pub struct AddBackward<T: FloatDType> {
    lhs: Tensor<T, DimDyn>,
    rhs: Tensor<T, DimDyn>,
}

impl<T: FloatDType> AddBackward<T> {
    pub fn new(lhs: Tensor<T, DimDyn>, rhs: Tensor<T, DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T: FloatDType> GradFn<T> for AddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        let grad_lhs = reduce_grad_for_broadcast_generic(grad_output, self.lhs.shape());
        let grad_rhs = reduce_grad_for_broadcast_generic(grad_output, self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient for Mul: z = a * b
/// ∂L/∂a = ∂L/∂z · b, ∂L/∂b = ∂L/∂z · a
pub struct MulBackward<T: FloatDType> {
    lhs: Tensor<T, DimDyn>,
    rhs: Tensor<T, DimDyn>,
}

impl<T: FloatDType> MulBackward<T> {
    pub fn new(lhs: Tensor<T, DimDyn>, rhs: Tensor<T, DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

// MulBackward is now fully generic since Mul<T> is generic over FloatDType.
// Gradient tensors have requires_grad() == false, so no new backward nodes are created.
impl<T: FloatDType> GradFn<T> for MulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        let grad_lhs_full = grad_output * &self.rhs;
        let grad_lhs = reduce_grad_for_broadcast_generic(&grad_lhs_full, self.lhs.shape());

        let grad_rhs_full = grad_output * &self.lhs;
        let grad_rhs = reduce_grad_for_broadcast_generic(&grad_rhs_full, self.rhs.shape());

        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient for Max: z = max(a, b)
/// ∂L/∂a = ∂L/∂z · (a ≥ b), ∂L/∂b = ∂L/∂z · (b > a)
pub struct MaxBackward<T: FloatDType> {
    lhs: Tensor<T, DimDyn>,
    rhs: Tensor<T, DimDyn>,
}

impl<T: FloatDType> MaxBackward<T> {
    pub fn new(lhs: Tensor<T, DimDyn>, rhs: Tensor<T, DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T: FloatDType> GradFn<T> for MaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Approximation: gradient flows to the larger input
        // TODO: Proper comparison operation needed
        let grad_lhs = reduce_grad_for_broadcast_generic(grad_output, self.lhs.shape());
        let grad_rhs = Tensor::<T, DimDyn>::zeros_dyn(self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

// ============================================================================
// Scalar operation backward functions
// ============================================================================

/// Gradient for scalar addition: z = a + c (c is scalar constant)
/// ∂L/∂a = ∂L/∂z (scalar addition passes gradient through unchanged)
pub struct ScalarAddBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
}

impl<T: FloatDType> ScalarAddBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType> GradFn<T> for ScalarAddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Gradient passes through unchanged for addition
        vec![grad_output.clone()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ScalarAddBackward"
    }
}

/// Gradient for scalar multiplication: z = a * c (c is scalar constant)
/// ∂L/∂a = c · ∂L/∂z (scalar multiplication scales the gradient)
pub struct ScalarMulBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    scalar: T,
}

impl<T: FloatDType> ScalarMulBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, scalar: T) -> Self {
        Self { input, scalar }
    }
}

impl<T: FloatDType> GradFn<T> for ScalarMulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Create a scalar tensor and multiply
        let scalar_tensor = Tensor::<T, DimDyn>::full_dyn(&[], self.scalar.clone());
        let grad_input = grad_output * &scalar_tensor;
        vec![grad_input]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ScalarMulBackward"
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Check if any input requires gradients (generic over FloatDType)
pub(crate) fn any_requires_grad_generic<T: FloatDType, D1: Dimension, D2: Dimension>(
    a: &Tensor<T, D1>,
    b: &Tensor<T, D2>,
) -> bool {
    a.requires_grad() || b.requires_grad()
}

/// Create a tensor with gradient tracking if needed (generic over FloatDType)
pub(crate) fn with_grad_fn_generic<T: FloatDType, D: Dimension>(
    tensor: Tensor<T, D>,
    grad_fn: Option<Arc<dyn GradFn<T>>>,
) -> Tensor<T, D> {
    if let Some(grad_fn) = grad_fn {
        // Create a new TensorInner with autograd metadata
        let inner = TensorInner {
            op: tensor.inner.op.clone(),
            view: tensor.inner.view.clone(),
            shape: tensor.inner.shape.clone(),
            dtype: tensor.inner.dtype.clone(),
            name: tensor.inner.name.clone(),
            autograd: Some(T::wrap_grad_fn(grad_fn)),
            buffer: std::sync::RwLock::new(None),
        };
        Tensor {
            inner: Arc::new(inner),
            autograd_typed: None,
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    } else {
        tensor
    }
}

/// Create a tensor with gradient tracking if needed (f32 specialized for backwards compatibility)
pub(crate) fn with_grad_fn<D: Dimension>(
    tensor: Tensor<f32, D>,
    grad_fn: Option<Arc<dyn GradFn<f32>>>,
) -> Tensor<f32, D> {
    with_grad_fn_generic(tensor, grad_fn)
}

/// Helper to compute result shape for binary operations with broadcasting
pub(crate) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_dim = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let b_dim = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            panic!(
                "Cannot broadcast shapes {:?} and {:?} at dimension {}",
                a, b, i
            );
        }
    }
    result
}

/// Helper to create View from usize shape
fn view_from_shape(shape: &[usize]) -> View {
    let shape_exprs: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
    View::contiguous(shape_exprs)
}

/// Create a binary elementwise Tensor using Compute variant
fn create_binary_elementwise<T: NumericDType, D: Dimension>(
    op: ElementwiseOp,
    lhs: &Tensor<T, D>,
    rhs: &Tensor<T, impl Dimension>,
) -> Tensor<T, D> {
    let result_shape = broadcast_shapes(lhs.shape(), rhs.shape());
    let view = view_from_shape(&result_shape);

    // Create Compute operation with inputs embedded
    let inputs = vec![lhs.as_input_ref(), rhs.as_input_ref()];
    let expr = op.to_ast(2);

    let inner = TensorInner::new(
        TensorOp::elementwise(inputs, expr),
        view,
        result_shape,
        T::DTYPE,
    );

    Tensor {
        inner: Arc::new(inner),
        autograd_typed: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

/// Create a tensor with scalar (constant fill) for binary ops (f32)
fn scalar_tensor_f32(value: f32) -> Tensor<f32, DimDyn> {
    // For scalar operations, we create a scalar (shape=[]) tensor
    // that will be broadcast to the target shape
    let view = View::contiguous(Vec::<Expr>::new()); // scalar
    let inner = TensorInner::new(
        TensorOp::ConstFill(Literal::F32(value)),
        view,
        vec![],
        DType::F32,
    );
    Tensor {
        inner: Arc::new(inner),
        autograd_typed: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

/// Create a tensor with scalar (constant fill) for binary ops (f64)
fn scalar_tensor_f64(value: f64) -> Tensor<f64, DimDyn> {
    let view = View::contiguous(Vec::<Expr>::new());
    let inner = TensorInner::new(
        TensorOp::ConstFill(Literal::F64(value)),
        view,
        vec![],
        DType::F64,
    );
    Tensor {
        inner: Arc::new(inner),
        autograd_typed: None,
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Add: Tensor + Tensor (generic over FloatDType with gradient tracking)
// ============================================================================

impl<T: FloatDType, D: Dimension> Add for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: Self) -> Tensor<T, D> {
        let result = create_binary_elementwise(ElementwiseOp::Add, self, rhs);
        // Use new typed system if any input has typed autograd
        if any_requires_grad_typed(self, rhs) {
            let grad_fn = AddBackwardTyped::new(self.clone(), rhs.clone());
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else if any_requires_grad_generic(self, rhs) {
            // Fallback to legacy system
            let grad_fn = AddBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Add<Tensor<T, D>> for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        self + &rhs
    }
}

impl<T: FloatDType, D: Dimension> Add<&Tensor<T, D>> for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        &self + rhs
    }
}

impl<T: FloatDType, D: Dimension> Add for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: Self) -> Tensor<T, D> {
        &self + &rhs
    }
}

// ============================================================================
// Mul: Tensor * Tensor (generic over FloatDType with gradient tracking)
// ============================================================================

impl<T: FloatDType, D: Dimension> Mul for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: Self) -> Tensor<T, D> {
        let result = create_binary_elementwise(ElementwiseOp::Mul, self, rhs);
        // Use new typed system if any input has typed autograd
        if any_requires_grad_typed(self, rhs) {
            let grad_fn = MulBackwardTyped::new(self.clone(), rhs.clone());
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else if any_requires_grad_generic(self, rhs) {
            // Fallback to legacy system
            let grad_fn = MulBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<T: FloatDType, D: Dimension> Mul<Tensor<T, D>> for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        self * &rhs
    }
}

impl<T: FloatDType, D: Dimension> Mul<&Tensor<T, D>> for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        &self * rhs
    }
}

impl<T: FloatDType, D: Dimension> Mul for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: Self) -> Tensor<T, D> {
        &self * &rhs
    }
}

// ============================================================================
// Scalar operations with gradient tracking (f32)
// ============================================================================

// Tensor + scalar (f32)
impl<D: Dimension> Add<f32> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: f32) -> Tensor<f32, D> {
        let scalar = scalar_tensor_f32(rhs);
        let result = create_binary_elementwise(ElementwiseOp::Add, self, &scalar);
        if self.requires_grad_typed() {
            let grad_fn = ScalarAddBackwardTyped::new(self.clone());
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else if self.requires_grad() {
            let grad_fn = ScalarAddBackward::new(self.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Add<f32> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: f32) -> Tensor<f32, D> {
        (&self).add(rhs)
    }
}

// scalar + Tensor (f32)
impl<D: Dimension> Add<&Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        rhs + self // commutative
    }
}

impl<D: Dimension> Add<Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self.add(&rhs)
    }
}

// Tensor * scalar (f32)
impl<D: Dimension> Mul<f32> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: f32) -> Tensor<f32, D> {
        let scalar = scalar_tensor_f32(rhs);
        let result = create_binary_elementwise(ElementwiseOp::Mul, self, &scalar);
        if self.requires_grad_typed() {
            let grad_fn = ScalarMulBackwardTyped::new(self.clone(), rhs);
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else if self.requires_grad() {
            let grad_fn = ScalarMulBackward::new(self.clone().into_dyn(), rhs);
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Mul<f32> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: f32) -> Tensor<f32, D> {
        (&self).mul(rhs)
    }
}

// scalar * Tensor (f32)
impl<D: Dimension> Mul<&Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        rhs * self // commutative
    }
}

impl<D: Dimension> Mul<Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self.mul(&rhs)
    }
}

// ============================================================================
// Scalar operations with gradient tracking (f64)
// ============================================================================

// Tensor + scalar (f64)
impl<D: Dimension> Add<f64> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: f64) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(rhs);
        let result = create_binary_elementwise(ElementwiseOp::Add, self, &scalar);
        if self.requires_grad_typed() {
            let grad_fn = ScalarAddBackwardTyped::new(self.clone());
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else if self.requires_grad() {
            let grad_fn = ScalarAddBackward::new(self.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Add<f64> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: f64) -> Tensor<f64, D> {
        (&self).add(rhs)
    }
}

// scalar + Tensor (f64)
impl<D: Dimension> Add<&Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        rhs + self // commutative
    }
}

impl<D: Dimension> Add<Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self.add(&rhs)
    }
}

// Tensor * scalar (f64)
impl<D: Dimension> Mul<f64> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: f64) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(rhs);
        let result = create_binary_elementwise(ElementwiseOp::Mul, self, &scalar);
        if self.requires_grad_typed() {
            let grad_fn = ScalarMulBackwardTyped::new(self.clone(), rhs);
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else if self.requires_grad() {
            let grad_fn = ScalarMulBackward::new(self.clone().into_dyn(), rhs);
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Mul<f64> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: f64) -> Tensor<f64, D> {
        (&self).mul(rhs)
    }
}

// scalar * Tensor (f64)
impl<D: Dimension> Mul<&Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        rhs * self // commutative
    }
}

impl<D: Dimension> Mul<Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self.mul(&rhs)
    }
}

// ============================================================================
// Max: element-wise maximum
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Compute element-wise maximum with another tensor (primop)
    pub fn maximum(&self, other: &Tensor<f32, D>) -> Tensor<f32, D> {
        let result = create_binary_elementwise(ElementwiseOp::Max, self, other);

        // Use typed system if available
        if self.requires_grad_typed() || other.requires_grad_typed() {
            let grad_fn = MaxBackwardTyped::new(self.clone(), other.clone());
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else if self.requires_grad() || other.requires_grad() {
            let grad_fn = MaxBackward::new(self.clone().into_dyn(), other.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute element-wise maximum with a scalar
    pub fn maximum_scalar(&self, value: f32) -> Tensor<f32, D> {
        let scalar = scalar_tensor_f32(value);
        let result = create_binary_elementwise(ElementwiseOp::Max, self, &scalar);

        // Scalar operations use legacy system (scalar is DimDyn)
        if self.requires_grad() || self.requires_grad_typed() {
            let grad_fn = MaxBackward::new(self.clone().into_dyn(), scalar);
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

// ============================================================================
// Max: Tensor<f64> (no gradient tracking)
// ============================================================================

impl<D: Dimension> Tensor<f64, D> {
    /// Compute element-wise maximum with another tensor (primop)
    pub fn maximum(&self, other: &Tensor<f64, D>) -> Tensor<f64, D> {
        let result = create_binary_elementwise(ElementwiseOp::Max, self, other);

        // Use typed system if available
        if self.requires_grad_typed() || other.requires_grad_typed() {
            let grad_fn = MaxBackwardTyped::new(self.clone(), other.clone());
            result.with_grad_fn_typed(Arc::new(grad_fn))
        } else {
            result
        }
    }

    /// Compute element-wise maximum with a scalar
    pub fn maximum_scalar(&self, value: f64) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(value);
        create_binary_elementwise(ElementwiseOp::Max, self, &scalar)
    }
}

// ============================================================================
// Integer operations: Idiv and Rem (IntegerDType only)
// ============================================================================

impl<T: IntegerDType, D: Dimension> Tensor<T, D> {
    /// Integer division (floor division)
    ///
    /// Computes element-wise integer division `self / other`.
    /// For negative dividends, this uses truncation toward zero (like Rust's `/` operator).
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 7);
    /// let b = Tensor::<i32, Dim2>::full([2, 3], 3);
    /// let c = a.idiv(&b); // Results in [[2, 2, 2], [2, 2, 2]]
    /// ```
    pub fn idiv(&self, other: &Tensor<T, impl Dimension>) -> Tensor<T, D> {
        create_binary_elementwise(ElementwiseOp::Idiv, self, other)
    }

    /// Remainder (modulo) operation
    ///
    /// Computes element-wise remainder `self % other`.
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::<i32, Dim2>::full([2, 3], 7);
    /// let b = Tensor::<i32, Dim2>::full([2, 3], 3);
    /// let c = a.rem(&b); // Results in [[1, 1, 1], [1, 1, 1]]
    /// ```
    pub fn rem(&self, other: &Tensor<T, impl Dimension>) -> Tensor<T, D> {
        create_binary_elementwise(ElementwiseOp::Rem, self, other)
    }
}

// ============================================================================
// Typed Backward Structs (new system with static dimension typing)
// ============================================================================

/// Typed gradient for Add: z = a + b (same dimension)
/// ∂L/∂a = ∂L/∂z, ∂L/∂b = ∂L/∂z
pub struct AddBackwardTyped<T: FloatDType, D: Dimension> {
    lhs: Tensor<T, D>,
    rhs: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> AddBackwardTyped<T, D> {
    pub fn new(lhs: Tensor<T, D>, rhs: Tensor<T, D>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for AddBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        // Propagate gradient directly to inputs (no broadcasting in this framework)
        if self.lhs.requires_grad_typed() {
            self.lhs.backward_with_typed(grad_output.clone());
        }
        if self.rhs.requires_grad_typed() {
            self.rhs.backward_with_typed(grad_output.clone());
        }
    }

    fn name(&self) -> &'static str {
        "AddBackwardTyped"
    }
}

/// Typed gradient for Mul: z = a * b (same dimension)
/// ∂L/∂a = ∂L/∂z · b, ∂L/∂b = ∂L/∂z · a
pub struct MulBackwardTyped<T: FloatDType, D: Dimension> {
    lhs: Tensor<T, D>,
    rhs: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> MulBackwardTyped<T, D> {
    pub fn new(lhs: Tensor<T, D>, rhs: Tensor<T, D>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for MulBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        // ∂L/∂a = ∂L/∂z · b
        if self.lhs.requires_grad_typed() {
            let grad_lhs = grad_output * &self.rhs;
            self.lhs.backward_with_typed(grad_lhs);
        }
        // ∂L/∂b = ∂L/∂z · a
        if self.rhs.requires_grad_typed() {
            let grad_rhs = grad_output * &self.lhs;
            self.rhs.backward_with_typed(grad_rhs);
        }
    }

    fn name(&self) -> &'static str {
        "MulBackwardTyped"
    }
}

/// Typed gradient for Max: z = max(a, b) (same dimension)
pub struct MaxBackwardTyped<T: FloatDType, D: Dimension> {
    lhs: Tensor<T, D>,
    rhs: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> MaxBackwardTyped<T, D> {
    pub fn new(lhs: Tensor<T, D>, rhs: Tensor<T, D>) -> Self {
        Self { lhs, rhs }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for MaxBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        // Approximation: gradient flows to the lhs (left input)
        // TODO: Proper comparison-based gradient routing
        if self.lhs.requires_grad_typed() {
            self.lhs.backward_with_typed(grad_output.clone());
        }
        // rhs gets zero gradient (approximation)
    }

    fn name(&self) -> &'static str {
        "MaxBackwardTyped"
    }
}

/// Typed gradient for scalar addition: z = a + c (c is scalar)
/// ∂L/∂a = ∂L/∂z
pub struct ScalarAddBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> ScalarAddBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for ScalarAddBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad_typed() {
            self.input.backward_with_typed(grad_output.clone());
        }
    }

    fn name(&self) -> &'static str {
        "ScalarAddBackwardTyped"
    }
}

/// Typed gradient for scalar multiplication: z = a * c (c is scalar)
/// ∂L/∂a = c · ∂L/∂z
pub struct ScalarMulBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    scalar: T,
}

impl<T: FloatDType, D: Dimension> ScalarMulBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, scalar: T) -> Self {
        Self { input, scalar }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for ScalarMulBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad_typed() {
            // Create scalar tensor and multiply
            let scalar_tensor = Tensor::<T, DimDyn>::full_dyn(&[], self.scalar.clone());
            // Convert to same dimension type
            let scalar_d = Tensor::<T, D> {
                inner: scalar_tensor.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_input = grad_output * &scalar_d;
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ScalarMulBackwardTyped"
    }
}

/// Check if any input requires typed gradients
pub(crate) fn any_requires_grad_typed<T: FloatDType, D: Dimension>(
    a: &Tensor<T, D>,
    b: &Tensor<T, D>,
) -> bool {
    a.requires_grad_typed() || b.requires_grad_typed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_add_tensors() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::<f32, Dim2>::zeros([2, 3]);
        let c = &a + 1.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_mul_tensors() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::full([2, 3], 2.0);
        let c = &a * &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_maximum() {
        let a = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let b = Tensor::<f32, Dim2>::full([2, 3], 0.0);
        let c = a.maximum(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_maximum_scalar() {
        let a = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let c = a.maximum_scalar(0.0);
        assert_eq!(c.shape(), &[2, 3]);
    }

    // f64 tests
    #[test]
    fn test_add_tensors_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = Tensor::<f64, Dim2>::ones([2, 3]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_add_scalar_f64() {
        let a = Tensor::<f64, Dim2>::zeros([2, 3]);
        let c = &a + 1.0f64;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_mul_tensors_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = Tensor::<f64, Dim2>::full([2, 3], 2.0);
        let c = &a * &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_maximum_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], -1.0);
        let b = Tensor::<f64, Dim2>::full([2, 3], 0.0);
        let c = a.maximum(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_maximum_scalar_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], -1.0);
        let c = a.maximum_scalar(0.0);
        assert_eq!(c.shape(), &[2, 3]);
    }

    // Integer operation tests
    #[test]
    fn test_idiv_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 7);
        let b = Tensor::<i32, Dim2>::full([2, 3], 3);
        let c = a.idiv(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_rem_i32() {
        let a = Tensor::<i32, Dim2>::full([2, 3], 7);
        let b = Tensor::<i32, Dim2>::full([2, 3], 3);
        let c = a.rem(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_idiv_u64() {
        let a = Tensor::<u64, Dim2>::full([2, 3], 10);
        let b = Tensor::<u64, Dim2>::full([2, 3], 3);
        let c = a.idiv(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_rem_u64() {
        let a = Tensor::<u64, Dim2>::full([2, 3], 10);
        let b = Tensor::<u64, Dim2>::full([2, 3], 3);
        let c = a.rem(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    // ========================================================================
    // Scalar operation gradient tests
    // ========================================================================

    #[test]
    fn test_scalar_add_backward() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let c = &a + 5.0;

        assert!(c.requires_grad());
        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_scalar_mul_backward() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let c = &a * 3.0;

        assert!(c.requires_grad());
        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_scalar_add_chain_backward() {
        // Test chained scalar operations: (a + 1) * 2
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = &a + 1.0;
        let c = &b * 2.0;

        assert!(c.requires_grad());
        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_scalar_add_backward_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);
        let c = &a + 5.0f64;

        assert!(c.requires_grad());
        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_scalar_mul_backward_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);
        let c = &a * 3.0f64;

        assert!(c.requires_grad());
        c.backward();

        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }
}
