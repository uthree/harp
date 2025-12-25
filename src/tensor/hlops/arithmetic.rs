//! Arithmetic high-level operations
//!
//! - Sub(a, b) = Add(a, Neg(b))
//! - Div(a, b) = Mul(a, Recip(b))

use std::ops::{Div, Sub};

use crate::tensor::{Dimension, Tensor};

// ============================================================================
// Sub: Tensor - Tensor = Add(a, Neg(b))
// ============================================================================

impl<D: Dimension> Sub for &Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: Self) -> Tensor<D> {
        // Sub(a, b) = Add(a, Neg(b))
        let neg_rhs = -rhs;
        self + &neg_rhs
    }
}

impl<D: Dimension> Sub<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;
    fn sub(self, rhs: Tensor<D>) -> Tensor<D> {
        self - &rhs
    }
}

impl<D: Dimension> Sub<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;
    fn sub(self, rhs: &Tensor<D>) -> Tensor<D> {
        &self - rhs
    }
}

impl<D: Dimension> Sub for Tensor<D> {
    type Output = Tensor<D>;
    fn sub(self, rhs: Self) -> Tensor<D> {
        &self - &rhs
    }
}

// Sub: Tensor - f32
impl<D: Dimension> Sub<f32> for &Tensor<D> {
    type Output = Tensor<D>;
    fn sub(self, rhs: f32) -> Tensor<D> {
        // a - c = a + (-c)
        self + (-rhs)
    }
}

impl<D: Dimension> Sub<f32> for Tensor<D> {
    type Output = Tensor<D>;
    fn sub(self, rhs: f32) -> Tensor<D> {
        &self - rhs
    }
}

// Sub: f32 - Tensor
impl<D: Dimension> Sub<&Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn sub(self, rhs: &Tensor<D>) -> Tensor<D> {
        // c - a = c + (-a)
        self + (-rhs)
    }
}

impl<D: Dimension> Sub<Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn sub(self, rhs: Tensor<D>) -> Tensor<D> {
        self - &rhs
    }
}

// ============================================================================
// Div: Tensor / Tensor = Mul(a, Recip(b))
// ============================================================================

#[allow(clippy::suspicious_arithmetic_impl)]
impl<D: Dimension> Div for &Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: Self) -> Tensor<D> {
        // Div(a, b) = Mul(a, Recip(b))
        let recip_rhs = rhs.recip();
        self * &recip_rhs
    }
}

impl<D: Dimension> Div<Tensor<D>> for &Tensor<D> {
    type Output = Tensor<D>;
    fn div(self, rhs: Tensor<D>) -> Tensor<D> {
        self / &rhs
    }
}

impl<D: Dimension> Div<&Tensor<D>> for Tensor<D> {
    type Output = Tensor<D>;
    fn div(self, rhs: &Tensor<D>) -> Tensor<D> {
        &self / rhs
    }
}

impl<D: Dimension> Div for Tensor<D> {
    type Output = Tensor<D>;
    fn div(self, rhs: Self) -> Tensor<D> {
        &self / &rhs
    }
}

// Div: Tensor / f32
impl<D: Dimension> Div<f32> for &Tensor<D> {
    type Output = Tensor<D>;
    fn div(self, rhs: f32) -> Tensor<D> {
        // a / c = a * (1/c)
        self * (1.0 / rhs)
    }
}

impl<D: Dimension> Div<f32> for Tensor<D> {
    type Output = Tensor<D>;
    fn div(self, rhs: f32) -> Tensor<D> {
        &self / rhs
    }
}

// Div: f32 / Tensor
#[allow(clippy::suspicious_arithmetic_impl)]
impl<D: Dimension> Div<&Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn div(self, rhs: &Tensor<D>) -> Tensor<D> {
        // c / a = c * recip(a)
        let recip = rhs.recip();
        self * &recip
    }
}

impl<D: Dimension> Div<Tensor<D>> for f32 {
    type Output = Tensor<D>;
    fn div(self, rhs: Tensor<D>) -> Tensor<D> {
        self / &rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_sub_tensors() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let b = Tensor::<Dim2>::ones([2, 3]);
        let c = &a - &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_sub_scalar() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = &a - 1.0;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_tensors() {
        let a = Tensor::<Dim2>::full([2, 3], 6.0);
        let b = Tensor::<Dim2>::full([2, 3], 2.0);
        let c = &a / &b;
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_div_scalar() {
        let a = Tensor::<Dim2>::full([2, 3], 6.0);
        let c = &a / 2.0;
        assert_eq!(c.shape(), &[2, 3]);
    }
}
