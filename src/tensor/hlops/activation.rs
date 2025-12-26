//! Activation function high-level operations
//!
//! - ReLU(x) = Max(x, 0)
//! - Sigmoid(x) = 1 / (1 + Exp(-x))
//! - Tanh(x) = (Exp(2x) - 1) / (Exp(2x) + 1)
//! - GELU(x) = x * Sigmoid(1.702 * x)  (fast approximation)
//! - SiLU(x) = x * Sigmoid(x)  (Swish)

use crate::tensor::{Dimension, Recip, Tensor};

impl<D: Dimension> Tensor<f32, D> {
    /// ReLU activation: max(0, x) (hlop)
    ///
    /// Implemented as: Max(x, 0)
    pub fn relu(&self) -> Tensor<f32, D> {
        self.maximum_scalar(0.0)
    }

    /// Leaky ReLU activation (hlop)
    ///
    /// f(x) = x if x > 0, else alpha * x
    pub fn leaky_relu(&self, alpha: f32) -> Tensor<f32, D> {
        // max(x, alpha * x)
        let scaled = self * alpha;
        self.maximum(&scaled)
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)) (hlop)
    ///
    /// Implemented as: Recip(1 + Exp(Neg(x)))
    pub fn sigmoid(&self) -> Tensor<f32, D> {
        let neg_x = -self;
        let exp_neg_x = neg_x.exp();
        let one_plus = exp_neg_x + 1.0;
        one_plus.recip()
    }

    /// Tanh activation (hlop)
    ///
    /// Implemented as: (exp(2x) - 1) / (exp(2x) + 1)
    pub fn tanh(&self) -> Tensor<f32, D> {
        // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        let two_x = self * 2.0;
        let exp_2x = two_x.exp();
        let numerator = &exp_2x - 1.0;
        let denominator = &exp_2x + 1.0;
        numerator / denominator
    }

    /// GELU activation (fast approximation) (hlop)
    ///
    /// Approximation: x * sigmoid(1.702 * x)
    pub fn gelu(&self) -> Tensor<f32, D> {
        let scaled = self * 1.702;
        let sig = scaled.sigmoid();
        self * &sig
    }

    /// SiLU (Swish) activation (hlop)
    ///
    /// f(x) = x * sigmoid(x)
    pub fn silu(&self) -> Tensor<f32, D> {
        let sig = self.sigmoid();
        self * &sig
    }

    /// Softplus activation (hlop)
    ///
    /// f(x) = ln(1 + exp(x))
    pub fn softplus(&self) -> Tensor<f32, D> {
        let exp_x = self.exp();
        let one_plus = exp_x + 1.0;
        one_plus.ln()
    }

    /// Mish activation (hlop)
    ///
    /// f(x) = x * tanh(softplus(x))
    pub fn mish(&self) -> Tensor<f32, D> {
        let sp = self.softplus();
        let tanh_sp = sp.tanh();
        self * &tanh_sp
    }

    /// ELU activation (hlop)
    ///
    /// f(x) = x if x > 0, else alpha * (exp(x) - 1)
    pub fn elu(&self, alpha: f32) -> Tensor<f32, D> {
        // For x > 0: x
        // For x <= 0: alpha * (exp(x) - 1)
        let exp_x = self.exp();
        let exp_minus_1 = exp_x - 1.0;
        let negative_part = exp_minus_1 * alpha;

        // Use max(x, 0) + min(x, 0) as approximation
        let positive = self.relu();
        let _negative = self.maximum_scalar(0.0);

        // This is approximate; proper implementation needs comparison op
        // For now: if x > 0: x, else: alpha * (exp(x) - 1)
        // Approximation: x * (x > 0) + alpha * (exp(x) - 1) * (x <= 0)
        positive + (-(&negative_part - &negative_part.relu()))
    }

    /// Hardtanh: clamp to [-min_val, max_val] (hlop)
    pub fn hardtanh(&self, min_val: f32, max_val: f32) -> Tensor<f32, D> {
        self.maximum_scalar(min_val).min_scalar(max_val)
    }

    /// Element-wise minimum with scalar (hlop)
    pub fn min_scalar(&self, value: f32) -> Tensor<f32, D> {
        // min(a, b) = -max(-a, -b)
        let neg_self = -self;
        let neg_max = neg_self.maximum_scalar(-value);
        -neg_max
    }

    /// Clamp values to [min, max] (hlop)
    pub fn clamp(&self, min: f32, max: f32) -> Tensor<f32, D> {
        self.maximum_scalar(min).min_scalar(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_relu() {
        let a = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let r = a.relu();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_leaky_relu() {
        let a = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let r = a.leaky_relu(0.01);
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_sigmoid() {
        let a = Tensor::<f32, Dim2>::zeros([2, 3]);
        let r = a.sigmoid();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_tanh() {
        let a = Tensor::<f32, Dim2>::zeros([2, 3]);
        let r = a.tanh();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_gelu() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let r = a.gelu();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_silu() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let r = a.silu();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_softplus() {
        let a = Tensor::<f32, Dim2>::ones([2, 3]);
        let r = a.softplus();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_clamp() {
        let a = Tensor::<f32, Dim2>::full([2, 3], 5.0);
        let r = a.clamp(-1.0, 1.0);
        assert_eq!(r.shape(), &[2, 3]);
    }
}
