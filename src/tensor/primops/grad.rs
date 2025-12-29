//! Gradient utility functions for primitive operations
//!
//! This module provides:
//! - Helper functions for gradient computation
//!
//! Individual typed operation gradients are defined in their respective modules:
//! - binary.rs: AddBackwardTyped, MulBackwardTyped, MaxBackwardTyped
//! - unary.rs: NegBackwardTyped, RecipBackwardTyped, SqrtBackwardTyped, etc.
//! - reduce.rs: SumBackwardTyped, ProdBackwardTyped, MaxReduceBackwardTyped
//!
//! The GradFnTyped trait is generic over T (FloatDType) and D (Dimension) for
//! statically-typed gradient computation.

use crate::tensor::{DimDyn, FloatDType, Tensor};

// ============================================================================
// Helper Functions
// ============================================================================

/// Reduce gradient to match the original input shape (handle broadcasting)
///
/// Generic version for any FloatDType (f32, f64).
pub fn reduce_grad_for_broadcast_generic<T: FloatDType>(
    grad: &Tensor<T, DimDyn>,
    target_shape: &[usize],
) -> Tensor<T, DimDyn> {
    if grad.shape() == target_shape {
        return grad.clone();
    }

    let grad_shape = grad.shape();
    let target_ndim = target_shape.len();
    let grad_ndim = grad_shape.len();

    // Pad target shape with 1s on the left to match grad_ndim
    let mut padded_target = vec![1usize; grad_ndim.saturating_sub(target_ndim)];
    padded_target.extend_from_slice(target_shape);

    // Find axes to reduce
    let mut reduce_axes = Vec::new();
    for (i, (&grad_dim, &target_dim)) in grad_shape.iter().zip(padded_target.iter()).enumerate() {
        if target_dim == 1 && grad_dim > 1 {
            reduce_axes.push(i);
        }
    }

    // Reduce along those axes using sum_axis (single axis at a time)
    // Process axes in reverse order to preserve indices
    let mut result = grad.clone();
    for &axis in reduce_axes.iter().rev() {
        // sum_axis reduces dimension, then unsqueeze restores it (keepdim=true effect)
        result = result.sum(axis).unsqueeze(axis);
    }

    // Reshape to target shape
    result.reshape_dyn(target_shape)
}

/// Reduce gradient to match the original input shape (handle broadcasting)
///
/// f32 specialized version for backwards compatibility.
pub fn reduce_grad_for_broadcast(
    grad: &Tensor<f32, DimDyn>,
    target_shape: &[usize],
) -> Tensor<f32, DimDyn> {
    reduce_grad_for_broadcast_generic(grad, target_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim1;
    use crate::tensor::primops::unary::Recip;

    #[test]
    fn test_reduce_grad_for_broadcast_same_shape() {
        let grad = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let result = reduce_grad_for_broadcast(&grad, &[2, 3]);
        assert_eq!(result.shape(), &[2, 3]);
    }

    // ========================================================================
    // f64 Autograd Integration Tests
    // ========================================================================

    #[test]
    fn test_f64_add_backward() {
        use crate::tensor::Dim2;

        // Create f64 tensors with gradient tracking
        let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);

        // Forward: c = a + b
        let c = &a + &b;
        assert!(c.requires_grad());
        assert_eq!(c.shape(), &[2, 3]);

        // Backward
        c.backward();

        // Check gradients: dc/da = 1, dc/db = 1
        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");

        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[2, 3]);
    }

    #[test]
    fn test_f64_mul_backward() {
        use crate::tensor::Dim2;

        // Create f64 tensors with gradient tracking
        let a = Tensor::<f64, Dim2>::full([2, 3], 2.0).set_requires_grad(true);
        let b = Tensor::<f64, Dim2>::full([2, 3], 3.0).set_requires_grad(true);

        // Forward: c = a * b
        let c = &a * &b;
        assert!(c.requires_grad());

        // Backward
        c.backward();

        // Check gradients: dc/da = b = 3, dc/db = a = 2
        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");

        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[2, 3]);
    }

    #[test]
    fn test_f64_neg_backward() {
        use crate::tensor::Dim2;

        // Create f64 tensor with gradient tracking
        let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);

        // Forward: b = -a
        let b = -&a;
        assert!(b.requires_grad());

        // Backward
        b.backward();

        // Check gradient: db/da = -1
        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_f64_recip_backward() {
        use crate::tensor::Dim2;

        // Create f64 tensor with gradient tracking (non-zero values)
        let a = Tensor::<f64, Dim2>::full([2, 3], 2.0).set_requires_grad(true);
        let a_clone = a.clone();

        // Forward: b = 1/a
        let b = a.recip();
        assert!(b.requires_grad());

        // Backward
        b.backward();

        // Check gradient: db/da = -1/a^2
        let grad_a = a_clone.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_f64_sub_backward() {
        use crate::tensor::Dim2;

        // Create f64 tensors with gradient tracking
        let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);

        // Forward: c = a - b (= a + (-b))
        let c = &a - &b;
        assert!(c.requires_grad());

        // Backward
        c.backward();

        // Check gradients: dc/da = 1, dc/db = -1
        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");

        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[2, 3]);
    }

    #[test]
    fn test_f64_div_backward() {
        use crate::tensor::Dim2;

        // Create f64 tensors with gradient tracking
        let a = Tensor::<f64, Dim2>::full([2, 3], 6.0).set_requires_grad(true);
        let b = Tensor::<f64, Dim2>::full([2, 3], 2.0).set_requires_grad(true);

        // Forward: c = a / b (= a * (1/b))
        let c = &a / &b;
        assert!(c.requires_grad());

        // Backward
        c.backward();

        // Check gradients exist
        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");

        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[2, 3]);
    }

    #[test]
    fn test_f64_chain_backward() {
        use crate::tensor::Dim2;

        // Create f64 tensors with gradient tracking
        let a = Tensor::<f64, Dim2>::full([2, 3], 2.0).set_requires_grad(true);
        let b = Tensor::<f64, Dim2>::full([2, 3], 3.0).set_requires_grad(true);

        // Forward: d = (a + b) * a = a^2 + ab
        let c = &a + &b;
        let d = &c * &a;
        assert!(d.requires_grad());

        // Backward
        d.backward();

        // Check gradients exist
        // dd/da = 2a + b, dd/db = a
        let grad_a = a.grad().expect("a should have gradient");
        let grad_b = b.grad().expect("b should have gradient");

        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[2, 3]);
    }

    #[test]
    fn test_f64_sum_axis_backward() {
        use crate::tensor::{Dim1, Dim2};

        // Create f64 tensor with gradient tracking
        let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);

        // Forward: b = sum_axis(a, 1)
        let b: Tensor<f64, Dim1> = a.sum(1);
        assert!(b.requires_grad());

        // Sum again to get scalar for backward
        let c = b.sum(0);
        c.backward();

        // Check gradient: dc/da = 1
        let grad_a = a.grad().expect("a should have gradient");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    // ========================================================================
    // Higher-order derivative tests
    // ========================================================================

    #[test]
    fn test_f32_backward_create_graph_builds_graph() {
        // Test that backward_create_graph creates gradients with requires_grad=true
        // y = x², dy/dx = 2x
        // The gradient 2x should have requires_grad=true when using backward_create_graph
        let x = Tensor::<f32, Dim1>::full([1], 3.0).set_requires_grad(true);

        // y = x * x = x²
        let y = &x * &x;

        // First backward with create_graph
        let _grad_y = y.backward_create_graph();

        // First derivative should exist
        let first_deriv = x
            .grad()
            .expect("x should have gradient after backward_create_graph");
        assert_eq!(first_deriv.shape(), &[1]);

        // The gradient should have requires_grad=true (key feature of backward_create_graph)
        // This allows computing higher-order derivatives
        assert!(
            first_deriv.requires_grad(),
            "Gradient from backward_create_graph should have requires_grad=true"
        );
    }

    #[test]
    fn test_f32_second_derivative_x_squared() {
        // Test: y = x², dy/dx = 2x, d²y/dx² = 2
        // At x = 3: dy/dx = 6, d²y/dx² = 2
        let x = Tensor::<f32, Dim1>::full([1], 3.0).set_requires_grad(true);

        // y = x * x = x²
        let y = &x * &x;

        // First backward with create_graph
        let _grad_y = y.backward_create_graph();

        // First derivative: dy/dx = 2x
        let first_deriv = x.grad().expect("x should have gradient");
        assert!(
            first_deriv.requires_grad(),
            "first_deriv should require grad"
        );

        // Reset x's gradient for second derivative computation
        x.zero_grad();

        // Second backward: differentiate the first derivative
        // first_deriv = 2x, so d(first_deriv)/dx = 2
        first_deriv.backward();

        // Second derivative should now be in x.grad()
        let second_deriv = x.grad().expect("x should have second derivative");
        assert_eq!(second_deriv.shape(), &[1]);
    }

    #[test]
    fn test_f64_backward_create_graph_builds_graph() {
        // Same test for f64
        let x = Tensor::<f64, Dim1>::full([1], 2.0).set_requires_grad(true);

        // y = x * x * x = x³
        let y = &(&x * &x) * &x;

        // First backward with create_graph
        let _grad_y = y.backward_create_graph();

        // First derivative should exist
        let first_deriv = x.grad().expect("x should have gradient");
        assert_eq!(first_deriv.shape(), &[1]);

        // The gradient should have requires_grad=true
        assert!(
            first_deriv.requires_grad(),
            "Gradient from backward_create_graph should have requires_grad=true"
        );
    }

    #[test]
    fn test_f32_create_graph_product() {
        // Test: y = x * z where both require gradients
        // dy/dx = z, dy/dz = x
        let x = Tensor::<f32, Dim1>::full([1], 2.0).set_requires_grad(true);
        let z = Tensor::<f32, Dim1>::full([1], 3.0).set_requires_grad(true);

        let y = &x * &z;

        // First backward with create_graph
        let _grad_y = y.backward_create_graph();

        // Both should have gradients
        let dx = x.grad().expect("x should have gradient");
        let dz = z.grad().expect("z should have gradient");

        // Both gradients should have requires_grad=true
        assert!(dx.requires_grad(), "dx should require grad");
        assert!(dz.requires_grad(), "dz should require grad");
    }
}
