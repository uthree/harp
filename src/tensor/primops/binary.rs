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
    DimDyn, Dimension, ElementwiseOp, FloatDType, FloatDTypeAutograd, GradFn, NumericDType, Tensor,
    TensorDType, TensorInner, TensorOp,
};

use super::grad::reduce_grad_for_broadcast_generic;

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

impl<T: FloatDTypeAutograd> GradFn<T> for AddBackward<T> {
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

// MulBackward is now fully generic since Mul<T> is generic over FloatDTypeAutograd.
// Gradient tensors have requires_grad() == false, so no new backward nodes are created.
impl<T: FloatDTypeAutograd> GradFn<T> for MulBackward<T> {
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

impl GradFn<f32> for MaxBackward<f32> {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // Approximation: gradient flows to the larger input
        // TODO: Proper comparison operation needed
        let grad_lhs = reduce_grad_for_broadcast_generic(grad_output, self.lhs.shape());
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

impl GradFn<f64> for MaxBackward<f64> {
    fn backward(&self, grad_output: &Tensor<f64, DimDyn>) -> Vec<Tensor<f64, DimDyn>> {
        // Approximation: gradient flows to the larger input
        // TODO: Proper comparison operation needed
        let grad_lhs = reduce_grad_for_broadcast_generic(grad_output, self.lhs.shape());
        let grad_rhs = Tensor::<f64, DimDyn>::zeros_dyn(self.rhs.shape());
        vec![grad_lhs, grad_rhs]
    }

    fn inputs(&self) -> Vec<Tensor<f64, DimDyn>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
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
pub(crate) fn with_grad_fn_generic<T: FloatDTypeAutograd, D: Dimension>(
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
// Add: Tensor + Tensor (generic over FloatDTypeAutograd with gradient tracking)
// ============================================================================

impl<T: FloatDTypeAutograd, D: Dimension> Add for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: Self) -> Tensor<T, D> {
        let result = create_binary_elementwise(ElementwiseOp::Add, self, rhs);
        if any_requires_grad_generic(self, rhs) {
            let grad_fn = AddBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<T: FloatDTypeAutograd, D: Dimension> Add<Tensor<T, D>> for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        self + &rhs
    }
}

impl<T: FloatDTypeAutograd, D: Dimension> Add<&Tensor<T, D>> for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        &self + rhs
    }
}

impl<T: FloatDTypeAutograd, D: Dimension> Add for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn add(self, rhs: Self) -> Tensor<T, D> {
        &self + &rhs
    }
}

// ============================================================================
// Mul: Tensor * Tensor (generic over FloatDTypeAutograd with gradient tracking)
// ============================================================================

impl<T: FloatDTypeAutograd, D: Dimension> Mul for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: Self) -> Tensor<T, D> {
        let result = create_binary_elementwise(ElementwiseOp::Mul, self, rhs);
        if any_requires_grad_generic(self, rhs) {
            let grad_fn = MulBackward::new(self.clone().into_dyn(), rhs.clone().into_dyn());
            with_grad_fn_generic(result, Some(Arc::new(grad_fn)))
        } else {
            result
        }
    }
}

impl<T: FloatDTypeAutograd, D: Dimension> Mul<Tensor<T, D>> for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        self * &rhs
    }
}

impl<T: FloatDTypeAutograd, D: Dimension> Mul<&Tensor<T, D>> for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        &self * rhs
    }
}

impl<T: FloatDTypeAutograd, D: Dimension> Mul for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn mul(self, rhs: Self) -> Tensor<T, D> {
        &self * &rhs
    }
}

// ============================================================================
// Macro for scalar operations (still needed because T cannot unify both element and scalar)
// ============================================================================

macro_rules! impl_scalar_ops {
    ($trait:ident, $method:ident, $op:expr, $T:ty, $scalar_fn:ident) => {
        // Tensor op scalar
        impl<D: Dimension> $trait<$T> for &Tensor<$T, D> {
            type Output = Tensor<$T, D>;
            fn $method(self, rhs: $T) -> Tensor<$T, D> {
                let scalar = $scalar_fn(rhs);
                create_binary_elementwise($op, self, &scalar)
            }
        }

        impl<D: Dimension> $trait<$T> for Tensor<$T, D> {
            type Output = Tensor<$T, D>;
            fn $method(self, rhs: $T) -> Tensor<$T, D> {
                (&self).$method(rhs)
            }
        }

        // scalar op Tensor
        impl<D: Dimension> $trait<&Tensor<$T, D>> for $T {
            type Output = Tensor<$T, D>;
            fn $method(self, rhs: &Tensor<$T, D>) -> Tensor<$T, D> {
                let scalar = $scalar_fn(self);
                create_binary_elementwise($op, rhs, &scalar)
            }
        }

        impl<D: Dimension> $trait<Tensor<$T, D>> for $T {
            type Output = Tensor<$T, D>;
            fn $method(self, rhs: Tensor<$T, D>) -> Tensor<$T, D> {
                self.$method(&rhs)
            }
        }
    };
}

// Scalar operations for Add
impl_scalar_ops!(Add, add, ElementwiseOp::Add, f32, scalar_tensor_f32);
impl_scalar_ops!(Add, add, ElementwiseOp::Add, f64, scalar_tensor_f64);

// Scalar operations for Mul
impl_scalar_ops!(Mul, mul, ElementwiseOp::Mul, f32, scalar_tensor_f32);
impl_scalar_ops!(Mul, mul, ElementwiseOp::Mul, f64, scalar_tensor_f64);

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
