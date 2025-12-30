//! Arithmetic high-level operations
//!
//! - Sub(a, b) = Add(a, Neg(b))
//! - Div(a, b) = Mul(a, Recip(b))

use std::ops::{Div, Sub};

use crate::tensor::{Dimension, FloatDType, Recip, Tensor};

// ============================================================================
// Sub: Tensor - Tensor = Add(a, Neg(b)) - Generic over FloatDType
// ============================================================================

impl<T: FloatDType, D: Dimension> Sub for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn sub(self, rhs: Self) -> Tensor<T, D> {
        // Sub(a, b) = Add(a, Neg(b))
        let neg_rhs = -rhs;
        self + &neg_rhs
    }
}

impl<T: FloatDType, D: Dimension> Sub<Tensor<T, D>> for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn sub(self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        self - &rhs
    }
}

impl<T: FloatDType, D: Dimension> Sub<&Tensor<T, D>> for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn sub(self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        &self - rhs
    }
}

impl<T: FloatDType, D: Dimension> Sub for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn sub(self, rhs: Self) -> Tensor<T, D> {
        &self - &rhs
    }
}

// Sub: Tensor - scalar (f32)
impl<D: Dimension> Sub<f32> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: f32) -> Tensor<f32, D> {
        // a - c = a + (-c)
        self + (-rhs)
    }
}

impl<D: Dimension> Sub<f32> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: f32) -> Tensor<f32, D> {
        &self - rhs
    }
}

// Sub: scalar - Tensor (f32)
impl<D: Dimension> Sub<&Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        // c - a = c + (-a)
        self + (-rhs)
    }
}

impl<D: Dimension> Sub<Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self - &rhs
    }
}

// Sub: Tensor - scalar (f64)
impl<D: Dimension> Sub<f64> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: f64) -> Tensor<f64, D> {
        // a - c = a + (-c)
        self + (-rhs)
    }
}

impl<D: Dimension> Sub<f64> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: f64) -> Tensor<f64, D> {
        &self - rhs
    }
}

// Sub: scalar - Tensor (f64)
impl<D: Dimension> Sub<&Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        // c - a = c + (-a)
        self + (-rhs)
    }
}

impl<D: Dimension> Sub<Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self - &rhs
    }
}

// ============================================================================
// Div: Tensor / Tensor = Mul(a, Recip(b)) - Generic over FloatDType
// ============================================================================

#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: FloatDType, D: Dimension> Div for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn div(self, rhs: Self) -> Tensor<T, D> {
        // Div(a, b) = Mul(a, Recip(b))
        let recip_rhs = rhs.recip();
        self * &recip_rhs
    }
}

impl<T: FloatDType, D: Dimension> Div<Tensor<T, D>> for &Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn div(self, rhs: Tensor<T, D>) -> Tensor<T, D> {
        self / &rhs
    }
}

impl<T: FloatDType, D: Dimension> Div<&Tensor<T, D>> for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn div(self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        &self / rhs
    }
}

impl<T: FloatDType, D: Dimension> Div for Tensor<T, D> {
    type Output = Tensor<T, D>;
    fn div(self, rhs: Self) -> Tensor<T, D> {
        &self / &rhs
    }
}

// Div: Tensor / scalar (f32)
impl<D: Dimension> Div<f32> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn div(self, rhs: f32) -> Tensor<f32, D> {
        // a / c = a * (1/c)
        self * (1.0 / rhs)
    }
}

impl<D: Dimension> Div<f32> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn div(self, rhs: f32) -> Tensor<f32, D> {
        &self / rhs
    }
}

// Div: scalar / Tensor (f32)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<D: Dimension> Div<&Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn div(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        // c / a = c * recip(a)
        let recip = rhs.recip();
        self * &recip
    }
}

impl<D: Dimension> Div<Tensor<f32, D>> for f32 {
    type Output = Tensor<f32, D>;
    fn div(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self / &rhs
    }
}

// Div: Tensor / scalar (f64)
impl<D: Dimension> Div<f64> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn div(self, rhs: f64) -> Tensor<f64, D> {
        // a / c = a * (1/c)
        self * (1.0 / rhs)
    }
}

impl<D: Dimension> Div<f64> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn div(self, rhs: f64) -> Tensor<f64, D> {
        &self / rhs
    }
}

// Div: scalar / Tensor (f64)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<D: Dimension> Div<&Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn div(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        // c / a = c * recip(a)
        let recip = rhs.recip();
        self * &recip
    }
}

impl<D: Dimension> Div<Tensor<f64, D>> for f64 {
    type Output = Tensor<f64, D>;
    fn div(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self / &rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_sub_tensors() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let b = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = &a - &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sub_scalar() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = &a - 1.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_tensors() {
        let a = Tensor::<f32, Dim2>::full([2, 3], 6.0);
        let b = Tensor::<f32, Dim2>::full([2, 3], 2.0);
        let c = &a / &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_scalar() {
        let a = Tensor::<f32, Dim2>::full([2, 3], 6.0);
        let c = &a / 2.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    // f64 tests
    #[test]
    fn test_sub_tensors_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let b = Tensor::<f64, Dim2>::ones([2, 3]);
        let c = &a - &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sub_scalar_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let c = &a - 1.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_tensors_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], 6.0);
        let b = Tensor::<f64, Dim2>::full([2, 3], 2.0);
        let c = &a / &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_scalar_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], 6.0);
        let c = &a / 2.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    // Gradient tracking tests
    #[test]
    fn test_sub_scalar_backward() {
        // a - c uses a + (-c), which should track gradients via ScalarAddBackward
        let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
        let b = &a - 1.0;
        assert!(
            b.requires_grad(),
            "Scalar subtraction should preserve requires_grad"
        );
        b.backward();
        let grad_a = a.grad().expect("a should have gradient after backward");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_scalar_backward() {
        // a / c uses a * (1/c), which should track gradients via ScalarMulBackward
        let a = Tensor::<f32, Dim2>::full([2, 3], 6.0).set_requires_grad(true);
        let b = &a / 2.0;
        assert!(
            b.requires_grad(),
            "Scalar division should preserve requires_grad"
        );
        b.backward();
        let grad_a = a.grad().expect("a should have gradient after backward");
        assert_eq!(grad_a.shape(), &[2, 3]);
    }
}
