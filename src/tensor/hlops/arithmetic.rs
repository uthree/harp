//! Arithmetic high-level operations
//!
//! - Sub(a, b) = Add(a, Neg(b))
//! - Div(a, b) = Mul(a, Recip(b))

use std::ops::{Div, Sub};

use crate::tensor::{Dimension, Recip, Tensor};

// ============================================================================
// Sub: Tensor - Tensor = Add(a, Neg(b))
// ============================================================================

impl<D: Dimension> Sub for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn sub(self, rhs: Self) -> Tensor<f32, D> {
        // Sub(a, b) = Add(a, Neg(b))
        let neg_rhs = -rhs;
        self + &neg_rhs
    }
}

impl<D: Dimension> Sub<Tensor<f32, D>> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self - &rhs
    }
}

impl<D: Dimension> Sub<&Tensor<f32, D>> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        &self - rhs
    }
}

impl<D: Dimension> Sub for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: Self) -> Tensor<f32, D> {
        &self - &rhs
    }
}

// Sub: Tensor - f32
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

// Sub: f32 - Tensor
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

// ============================================================================
// Sub: Tensor<f64> - Tensor<f64> = Add(a, Neg(b))
// ============================================================================

impl<D: Dimension> Sub for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn sub(self, rhs: Self) -> Tensor<f64, D> {
        // Sub(a, b) = Add(a, Neg(b))
        let neg_rhs = -rhs;
        self + &neg_rhs
    }
}

impl<D: Dimension> Sub<Tensor<f64, D>> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self - &rhs
    }
}

impl<D: Dimension> Sub<&Tensor<f64, D>> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        &self - rhs
    }
}

impl<D: Dimension> Sub for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: Self) -> Tensor<f64, D> {
        &self - &rhs
    }
}

// Sub: Tensor - f64
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

// Sub: f64 - Tensor
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
// Div: Tensor / Tensor = Mul(a, Recip(b))
// ============================================================================

#[allow(clippy::suspicious_arithmetic_impl)]
impl<D: Dimension> Div for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;

    fn div(self, rhs: Self) -> Tensor<f32, D> {
        // Div(a, b) = Mul(a, Recip(b))
        let recip_rhs = rhs.recip();
        self * &recip_rhs
    }
}

impl<D: Dimension> Div<Tensor<f32, D>> for &Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn div(self, rhs: Tensor<f32, D>) -> Tensor<f32, D> {
        self / &rhs
    }
}

impl<D: Dimension> Div<&Tensor<f32, D>> for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn div(self, rhs: &Tensor<f32, D>) -> Tensor<f32, D> {
        &self / rhs
    }
}

impl<D: Dimension> Div for Tensor<f32, D> {
    type Output = Tensor<f32, D>;
    fn div(self, rhs: Self) -> Tensor<f32, D> {
        &self / &rhs
    }
}

// Div: Tensor / f32
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

// Div: f32 / Tensor
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

// ============================================================================
// Div: Tensor<f64> / Tensor<f64> = Mul(a, Recip(b))
// ============================================================================

#[allow(clippy::suspicious_arithmetic_impl)]
impl<D: Dimension> Div for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;

    fn div(self, rhs: Self) -> Tensor<f64, D> {
        // Div(a, b) = Mul(a, Recip(b))
        let recip_rhs = rhs.recip();
        self * &recip_rhs
    }
}

impl<D: Dimension> Div<Tensor<f64, D>> for &Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn div(self, rhs: Tensor<f64, D>) -> Tensor<f64, D> {
        self / &rhs
    }
}

impl<D: Dimension> Div<&Tensor<f64, D>> for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn div(self, rhs: &Tensor<f64, D>) -> Tensor<f64, D> {
        &self / rhs
    }
}

impl<D: Dimension> Div for Tensor<f64, D> {
    type Output = Tensor<f64, D>;
    fn div(self, rhs: Self) -> Tensor<f64, D> {
        &self / &rhs
    }
}

// Div: Tensor<f64> / f64
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

// Div: f64 / Tensor<f64>
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
}
