//! Transcendental high-level operations
//!
//! - Exp(x) = Exp2(x * log2(e))
//! - Ln(x) = Log2(x) * ln(2)
//! - Cos(x) = Sin(x + π/2)

use crate::tensor::{Dimension, Tensor};

impl<D: Dimension> Tensor<D> {
    /// Compute exp(x) = e^x for each element (hlop)
    ///
    /// Implemented as: Exp2(x * log2(e))
    pub fn exp(&self) -> Tensor<D> {
        // exp(x) = 2^(x * log2(e))
        let log2_e = std::f32::consts::LOG2_E;
        let scaled = self * log2_e;
        scaled.exp2()
    }

    /// Compute natural logarithm ln(x) for each element (hlop)
    ///
    /// Implemented as: Log2(x) * ln(2)
    pub fn ln(&self) -> Tensor<D> {
        // ln(x) = log2(x) * ln(2)
        let ln2 = std::f32::consts::LN_2;
        let log2_x = self.log2();
        log2_x * ln2
    }

    /// Compute cos(x) for each element (hlop)
    ///
    /// Implemented as: Sin(x + π/2)
    pub fn cos(&self) -> Tensor<D> {
        use std::f32::consts::FRAC_PI_2;
        let shifted = self + FRAC_PI_2;
        shifted.sin()
    }

    /// Compute tan(x) for each element (hlop)
    ///
    /// Implemented as: Sin(x) / Cos(x)
    pub fn tan(&self) -> Tensor<D> {
        let sin_x = self.sin();
        let cos_x = self.cos();
        sin_x / cos_x
    }

    /// Compute x^n for each element (hlop)
    ///
    /// Implemented as: Exp2(n * Log2(x)) for positive x
    pub fn pow(&self, n: f32) -> Tensor<D> {
        // x^n = 2^(n * log2(x))
        let log2_x = self.log2();
        let scaled = log2_x * n;
        scaled.exp2()
    }

    /// Compute x^2 for each element (hlop)
    pub fn square(&self) -> Tensor<D> {
        self * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_exp() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.exp();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_ln() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.ln();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_cos() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.cos();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_tan() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let c = a.tan();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_pow() {
        let a = Tensor::<Dim2>::full([2, 3], 2.0);
        let c = a.pow(3.0);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_square() {
        let a = Tensor::<Dim2>::full([2, 3], 3.0);
        let c = a.square();
        assert_eq!(c.shape(), &[2, 3]);
    }
}
