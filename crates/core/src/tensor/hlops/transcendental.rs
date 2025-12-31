//! Transcendental high-level operations
//!
//! These operations are available for FloatDType (f32, f64).
//! The underlying primops (sin, sqrt, log2, exp2) support both f32 and f64.
//!
//! - Exp(x) = Exp2(x * log2(e))
//! - Ln(x) = Log2(x) * ln(2)
//! - Cos(x) = Sin(x + π/2)

use crate::tensor::{Dimension, Exp2, FloatDType, Log2, Sin, Tensor};

impl<T: FloatDType, D: Dimension> Tensor<T, D> {
    /// Compute exp(x) = e^x for each element (hlop)
    ///
    /// Implemented as: Exp2(x * log2(e))
    pub fn exp(&self) -> Tensor<T, D> {
        let scaled = self * T::LOG2_E;
        scaled.exp2()
    }

    /// Compute natural logarithm ln(x) for each element (hlop)
    ///
    /// Implemented as: Log2(x) * ln(2)
    pub fn ln(&self) -> Tensor<T, D> {
        let log2_x = self.log2();
        log2_x * T::LN_2
    }

    /// Compute cos(x) for each element (hlop)
    ///
    /// Implemented as: Sin(x + π/2)
    pub fn cos(&self) -> Tensor<T, D> {
        let shifted = self + T::FRAC_PI_2;
        shifted.sin()
    }

    /// Compute tan(x) for each element (hlop)
    ///
    /// Implemented as: Sin(x) / Cos(x)
    pub fn tan(&self) -> Tensor<T, D> {
        let sin_x = self.sin();
        let cos_x = self.cos();
        sin_x / cos_x
    }

    /// Compute x^n for each element (hlop)
    ///
    /// Implemented as: Exp2(n * Log2(x)) for positive x
    pub fn pow(&self, n: T) -> Tensor<T, D> {
        let log2_x = self.log2();
        let scaled = log2_x * n;
        scaled.exp2()
    }

    /// Compute x^2 for each element (hlop)
    pub fn square(&self) -> Tensor<T, D> {
        self * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_exp_f32() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.exp();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_exp_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let c = a.exp();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_ln_f32() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.ln();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_ln_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let c = a.ln();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_cos_f32() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.cos();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_cos_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let c = a.cos();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_tan_f32() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let c = a.tan();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_tan_f64() {
        let a = Tensor::<f64, Dim2>::ones([2, 3]);
        let c = a.tan();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_pow_f32() {
        let a = Tensor::<f32, Dim2>::full([2, 3], 2.0);
        let c = a.pow(3.0);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_pow_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], 2.0);
        let c = a.pow(3.0);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_square_f32() {
        let a = Tensor::<f32, Dim2>::full([2, 3], 3.0);
        let c = a.square();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_square_f64() {
        let a = Tensor::<f64, Dim2>::full([2, 3], 3.0);
        let c = a.square();
        assert_eq!(c.shape(), &[2, 3]);
    }
}
