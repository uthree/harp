//! Binary primitive operations
//!
//! These operations support NumericDType (f32, f64, integers).
//! Gradient tracking is only available for f32 tensors.
//!
//! - Add: element-wise addition
//! - Mul: element-wise multiplication
//! - Max: element-wise maximum

use std::marker::PhantomData;
use std::ops::Add;
use std::ops::Mul;
use std::sync::{Arc, RwLock};

use crate::ast::{DType, Literal};
use crate::tensor::shape::{Expr, View};
use crate::tensor::{
    AutogradMeta, DimDyn, Dimension, ElementwiseOp, GradFn, NumericDType, Tensor, TensorDType,
    TensorInner, TensorOp,
};

use super::grad::reduce_grad_for_broadcast;

// ============================================================================
// Helper for type conversion
// ============================================================================

/// Convert any Tensor<T, D> to Tensor<f32, DimDyn> for graph operations.
fn to_graph_ref<T: TensorDType, D: Dimension>(tensor: &Tensor<T, D>) -> Tensor<f32, DimDyn> {
    Tensor {
        inner: tensor.inner.clone(),
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Binary Gradients (f32 only)
// ============================================================================

/// Gradient for Add: z = a + b
/// ∂L/∂a = ∂L/∂z, ∂L/∂b = ∂L/∂z
pub struct AddBackward {
    lhs: Tensor<f32, DimDyn>,
    rhs: Tensor<f32, DimDyn>,
}

impl AddBackward {
    pub fn new(lhs: Tensor<f32, DimDyn>, rhs: Tensor<f32, DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        let grad_lhs = reduce_grad_for_broadcast(grad_output, self.lhs.shape());
        let grad_rhs = reduce_grad_for_broadcast(grad_output, self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient for Mul: z = a * b
/// ∂L/∂a = ∂L/∂z · b, ∂L/∂b = ∂L/∂z · a
pub struct MulBackward {
    lhs: Tensor<f32, DimDyn>,
    rhs: Tensor<f32, DimDyn>,
}

impl MulBackward {
    pub fn new(lhs: Tensor<f32, DimDyn>, rhs: Tensor<f32, DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        let grad_lhs_full = grad_output * &self.rhs;
        let grad_lhs = reduce_grad_for_broadcast(&grad_lhs_full, self.lhs.shape());

        let grad_rhs_full = grad_output * &self.lhs;
        let grad_rhs = reduce_grad_for_broadcast(&grad_rhs_full, self.rhs.shape());

        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient for Max: z = max(a, b)
/// ∂L/∂a = ∂L/∂z · (a ≥ b), ∂L/∂b = ∂L/∂z · (b > a)
pub struct MaxBackward {
    lhs: Tensor<f32, DimDyn>,
    rhs: Tensor<f32, DimDyn>,
}

impl MaxBackward {
    pub fn new(lhs: Tensor<f32, DimDyn>, rhs: Tensor<f32, DimDyn>) -> Self {
        Self { lhs, rhs }
    }
}

impl GradFn for MaxBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // Approximation: gradient flows to the larger input
        // TODO: Proper comparison operation needed
        let grad_lhs = reduce_grad_for_broadcast(grad_output, self.lhs.shape());
        let grad_rhs = Tensor::<f32, DimDyn>::zeros_dyn(self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Check if any input requires gradients
pub(crate) fn any_requires_grad<D1: Dimension, D2: Dimension>(
    a: &Tensor<f32, D1>,
    b: &Tensor<f32, D2>,
) -> bool {
    a.requires_grad() || b.requires_grad()
}

/// Create a tensor with gradient tracking if needed
pub(crate) fn with_grad_fn<D: Dimension>(
    tensor: Tensor<f32, D>,
    grad_fn: Option<Arc<dyn GradFn>>,
) -> Tensor<f32, D> {
    if grad_fn.is_some() {
        // Create a new TensorInner with autograd metadata
        let inner = TensorInner {
            op: tensor.inner.op.clone(),
            view: tensor.inner.view.clone(),
            shape: tensor.inner.shape.clone(),
            dtype: tensor.inner.dtype.clone(),
            name: tensor.inner.name.clone(),
            autograd: Some(AutogradMeta {
                grad: RwLock::new(None),
                grad_fn,
            }),
            buffer: RwLock::new(None),
        };
        Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        }
    } else {
        tensor
    }
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
    let inputs = vec![Arc::new(to_graph_ref(lhs)), Arc::new(to_graph_ref(rhs))];
    let expr = op.to_ast(2);

    let inner = TensorInner::new(
        TensorOp::elementwise(inputs, expr),
        view,
        result_shape,
        T::DTYPE,
    );

    Tensor {
        inner: Arc::new(inner),
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
        _dtype: PhantomData,
        _dim: PhantomData,
    }
}

// ============================================================================
// Add: Tensor + Tensor
// ============================================================================

impl<D: Dimension> Add for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn add(self, rhs: Self) -> Tensor<f32, D> {
        let result = create_binary_elementwise(ElementwiseOp::Add, self, rhs);

        if any_requires_grad(self, rhs) {
            let grad_fn = AddBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Add<Tensor<f32, D>> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self + &rhs
    }
}

impl<D: Dimension> Add<&Tensor<f32, D>> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        &self + rhs
    }
}

impl<D: Dimension> Add for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: Self) -> Tensor<f32, D> {
        &self + &rhs
    }
}

// Add: Tensor + f32
impl<D: Dimension> Add<f32> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: f32) -> Tensor<f32, D> {
        let scalar = scalar_tensor_f32(rhs);
        create_binary_elementwise(ElementwiseOp::Add, self, &scalar)
    }
}

impl<D: Dimension> Add<f32> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: f32) -> Tensor<f32, D> {
        &self + rhs
    }
}

// Add: f32 + Tensor
impl<D: Dimension> Add<&Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        // Swap order: rhs determines the dimension type
        let scalar = scalar_tensor_f32(self);
        create_binary_elementwise(ElementwiseOp::Add, rhs, &scalar)
    }
}

impl<D: Dimension> Add<Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn add(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self + &rhs
    }
}

// ============================================================================
// Add: Tensor<f64> + Tensor<f64> (no gradient tracking)
// ============================================================================

impl<D: Dimension> Add for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn add(self, rhs: Self) -> Tensor<f64, D> {
        create_binary_elementwise(ElementwiseOp::Add, self, rhs)
    }
}

impl<D: Dimension> Add<Tensor<f64, D>> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self + &rhs
    }
}

impl<D: Dimension> Add<&Tensor<f64, D>> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        &self + rhs
    }
}

impl<D: Dimension> Add for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: Self) -> Tensor<f64, D> {
        &self + &rhs
    }
}

// Add: Tensor<f64> + f64
impl<D: Dimension> Add<f64> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: f64) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(rhs);
        create_binary_elementwise(ElementwiseOp::Add, self, &scalar)
    }
}

impl<D: Dimension> Add<f64> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: f64) -> Tensor<f64, D> {
        &self + rhs
    }
}

// Add: f64 + Tensor<f64>
impl<D: Dimension> Add<&Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(self);
        create_binary_elementwise(ElementwiseOp::Add, rhs, &scalar)
    }
}

impl<D: Dimension> Add<Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn add(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self + &rhs
    }
}

// ============================================================================
// Mul: Tensor * Tensor
// ============================================================================

impl<D: Dimension> Mul for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn mul(self, rhs: Self) -> Tensor<f32, D> {
        let result = create_binary_elementwise(ElementwiseOp::Mul, self, rhs);

        if any_requires_grad(self, rhs) {
            let grad_fn = MulBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<D: Dimension> Mul<Tensor<f32, D>> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self * &rhs
    }
}

impl<D: Dimension> Mul<&Tensor<f32, D>> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        &self * rhs
    }
}

impl<D: Dimension> Mul for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: Self) -> Tensor<f32, D> {
        &self * &rhs
    }
}

// Mul: Tensor * f32
impl<D: Dimension> Mul<f32> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: f32) -> Tensor<f32, D> {
        let scalar = scalar_tensor_f32(rhs);
        create_binary_elementwise(ElementwiseOp::Mul, self, &scalar)
    }
}

impl<D: Dimension> Mul<f32> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: f32) -> Tensor<f32, D> {
        &self * rhs
    }
}

// Mul: f32 * Tensor
impl<D: Dimension> Mul<&Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        // Swap order: rhs determines the dimension type
        let scalar = scalar_tensor_f32(self);
        create_binary_elementwise(ElementwiseOp::Mul, rhs, &scalar)
    }
}

impl<D: Dimension> Mul<Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self * &rhs
    }
}

// ============================================================================
// Mul: Tensor<f64> * Tensor<f64> (no gradient tracking)
// ============================================================================

impl<D: Dimension> Mul for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn mul(self, rhs: Self) -> Tensor<f64, D> {
        create_binary_elementwise(ElementwiseOp::Mul, self, rhs)
    }
}

impl<D: Dimension> Mul<Tensor<f64, D>> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self * &rhs
    }
}

impl<D: Dimension> Mul<&Tensor<f64, D>> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        &self * rhs
    }
}

impl<D: Dimension> Mul for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: Self) -> Tensor<f64, D> {
        &self * &rhs
    }
}

// Mul: Tensor<f64> * f64
impl<D: Dimension> Mul<f64> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: f64) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(rhs);
        create_binary_elementwise(ElementwiseOp::Mul, self, &scalar)
    }
}

impl<D: Dimension> Mul<f64> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: f64) -> Tensor<f64, D> {
        &self * rhs
    }
}

// Mul: f64 * Tensor<f64>
impl<D: Dimension> Mul<&Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(self);
        create_binary_elementwise(ElementwiseOp::Mul, rhs, &scalar)
    }
}

impl<D: Dimension> Mul<Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self * &rhs
    }
}

// ============================================================================
// Max: element-wise maximum
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Compute element-wise maximum with another tensor (primop)
    pub fn max(&self, other: &Tensor<f32, impl Dimension>) -> Tensor<f32, D> {
        let result = create_binary_elementwise(ElementwiseOp::Max, self, other);

        if self.requires_grad() || other.requires_grad() {
            let grad_fn = MaxBackward::new(self.clone().into_dyn(), other.clone().into_dyn());
            with_grad_fn(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }

    /// Compute element-wise maximum with a scalar
    pub fn max_scalar(&self, value: f32) -> Tensor<f32, D> {
        let scalar = scalar_tensor_f32(value);
        self.max(&scalar)
    }
}

// ============================================================================
// Max: Tensor<f64> (no gradient tracking)
// ============================================================================

impl<D: Dimension> Tensor<f64, D> {
    /// Compute element-wise maximum with another tensor (primop)
    pub fn max(&self, other: &Tensor<f64, impl Dimension>) -> Tensor<f64, D> {
        create_binary_elementwise(ElementwiseOp::Max, self, other)
    }

    /// Compute element-wise maximum with a scalar
    pub fn max_scalar(&self, value: f64) -> Tensor<f64, D> {
        let scalar = scalar_tensor_f64(value);
        self.max(&scalar)
    }
}

// ============================================================================
// Idiv: integer division (floor division)
// TODO: Requires floor() support in graph ops
// ============================================================================

// impl<D: Dimension> Tensor<f32, D> {
//     /// Integer division (floor division)
//     pub fn idiv(&self, other: &Tensor<f32, impl Dimension>) -> Tensor<f32, D> {
//         // Requires floor() primop support
//         todo!("idiv requires floor() support in graph ops")
//     }
// }

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
    fn test_max() {
        let a = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let b = Tensor::<f32, Dim2>::full([2, 3], 0.0);
        let c = a.max(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_max_scalar() {
        let a = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let c = a.max_scalar(0.0);
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
    fn test_max_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], -1.0);
        let b = Tensor::<f64, Dim2>::full([2, 3], 0.0);
        let c = a.max(&b);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_max_scalar_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], -1.0);
        let c = a.max_scalar(0.0);
        assert_eq!(c.shape(), &[2, 3]);
    }
}
