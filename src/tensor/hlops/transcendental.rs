//! Transcendental high-level operations
//!
//! These operations are available for FloatDType (f32, f64).
//! The underlying primops (sin, sqrt, log2, exp2) support both f32 and f64.
//!
//! - Exp(x) = Exp2(x * log2(e))
//! - Ln(x) = Log2(x) * ln(2)
//! - Cos(x) = Sin(x + π/2)

use crate::tensor::{Dimension, Exp2, FloatDType, Log2, Sin, Tensor};

mod sealed {
    use super::*;

    /// Sealed trait for transcendental operations on tensors
    pub trait TranscendentalOps<D: Dimension>: Sized {
        type Scalar;

        /// Compute exp(x) = e^x for each element
        fn exp(&self) -> Self;

        /// Compute natural logarithm ln(x) for each element
        fn ln(&self) -> Self;

        /// Compute cos(x) for each element
        fn cos(&self) -> Self;

        /// Compute tan(x) for each element
        fn tan(&self) -> Self;

        /// Compute x^n for each element
        fn pow(&self, n: Self::Scalar) -> Self;

        /// Compute x^2 for each element
        fn square(&self) -> Self;
    }

    impl<D: Dimension> TranscendentalOps<D> for Tensor<f32, D> {
        type Scalar = f32;

        fn exp(&self) -> Self {
            let scaled = self * f32::LOG2_E;
            scaled.exp2()
        }

        fn ln(&self) -> Self {
            let log2_x = self.log2();
            log2_x * f32::LN_2
        }

        fn cos(&self) -> Self {
            let shifted = self + f32::FRAC_PI_2;
            shifted.sin()
        }

        fn tan(&self) -> Self {
            let sin_x = self.sin();
            let cos_x = TranscendentalOps::cos(self);
            sin_x / cos_x
        }

        fn pow(&self, n: f32) -> Self {
            let log2_x = self.log2();
            let scaled = log2_x * n;
            scaled.exp2()
        }

        fn square(&self) -> Self {
            self * self
        }
    }

    impl<D: Dimension> TranscendentalOps<D> for Tensor<f64, D> {
        type Scalar = f64;

        fn exp(&self) -> Self {
            let scaled = self * f64::LOG2_E;
            scaled.exp2()
        }

        fn ln(&self) -> Self {
            let log2_x = self.log2();
            log2_x * f64::LN_2
        }

        fn cos(&self) -> Self {
            let shifted = self + f64::FRAC_PI_2;
            shifted.sin()
        }

        fn tan(&self) -> Self {
            let sin_x = self.sin();
            let cos_x = TranscendentalOps::cos(self);
            sin_x / cos_x
        }

        fn pow(&self, n: f64) -> Self {
            let log2_x = self.log2();
            let scaled = log2_x * n;
            scaled.exp2()
        }

        fn square(&self) -> Self {
            self * self
        }
    }
}

use sealed::TranscendentalOps;

impl<T: FloatDType, D: Dimension> Tensor<T, D>
where
    Tensor<T, D>: TranscendentalOps<D, Scalar = T>,
{
    /// Compute exp(x) = e^x for each element (hlop)
    ///
    /// Implemented as: Exp2(x * log2(e))
    pub fn exp(&self) -> Tensor<T, D> {
        TranscendentalOps::exp(self)
    }

    /// Compute natural logarithm ln(x) for each element (hlop)
    ///
    /// Implemented as: Log2(x) * ln(2)
    pub fn ln(&self) -> Tensor<T, D> {
        TranscendentalOps::ln(self)
    }

    /// Compute cos(x) for each element (hlop)
    ///
    /// Implemented as: Sin(x + π/2)
    pub fn cos(&self) -> Tensor<T, D> {
        TranscendentalOps::cos(self)
    }

    /// Compute tan(x) for each element (hlop)
    ///
    /// Implemented as: Sin(x) / Cos(x)
    pub fn tan(&self) -> Tensor<T, D> {
        TranscendentalOps::tan(self)
    }

    /// Compute x^n for each element (hlop)
    ///
    /// Implemented as: Exp2(n * Log2(x)) for positive x
    pub fn pow(&self, n: T) -> Tensor<T, D> {
        TranscendentalOps::pow(self, n)
    }

    /// Compute x^2 for each element (hlop)
    pub fn square(&self) -> Tensor<T, D> {
        TranscendentalOps::square(self)
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
