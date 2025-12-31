//! 活性化関数
//!
//! ニューラルネットワークで使用される活性化関数を提供します。
//!
//! # 使い方
//!
//! `ActivationExt`トレイトをインポートすると、テンソルに活性化関数メソッドが追加されます。
//!
//! ```ignore
//! use harp_nn::functional::activation::ActivationExt;
//!
//! let input = Tensor::<f32, Dim2>::ones([2, 3]);
//! let output = input.relu();
//! ```
//!
//! # 関数一覧
//!
//! - [`relu`](ActivationExt::relu) - ReLU: `max(0, x)`
//! - [`leaky_relu`](ActivationExt::leaky_relu) - Leaky ReLU: `max(alpha * x, x)`
//! - [`sigmoid`](ActivationExt::sigmoid) - Sigmoid: `1 / (1 + exp(-x))`
//! - [`tanh_activation`](ActivationExt::tanh_activation) - Tanh: `(exp(2x) - 1) / (exp(2x) + 1)`
//! - [`gelu`](ActivationExt::gelu) - GELU (高速近似): `x * sigmoid(1.702 * x)`
//! - [`silu`](ActivationExt::silu) - SiLU (Swish): `x * sigmoid(x)`
//! - [`softplus`](ActivationExt::softplus) - Softplus: `ln(1 + exp(x))`
//! - [`mish`](ActivationExt::mish) - Mish: `x * tanh(softplus(x))`
//! - [`elu`](ActivationExt::elu) - ELU: `x if x > 0, else alpha * (exp(x) - 1)`

use harp::tensor::{Dimension, FloatDType, Tensor};

/// 活性化関数の拡張トレイト
///
/// このトレイトをインポートすると、テンソルに活性化関数メソッドが追加されます。
/// 実装はharp-coreのTensorメソッドに委譲されます。
pub trait ActivationExt<D: Dimension>: Sized {
    /// スカラー型
    type Scalar: FloatDType;

    /// ReLU: `max(0, x)`
    fn relu(&self) -> Self;

    /// Leaky ReLU: `max(alpha * x, x)`
    fn leaky_relu(&self, alpha: Self::Scalar) -> Self;

    /// Sigmoid: `1 / (1 + exp(-x))`
    fn sigmoid(&self) -> Self;

    /// Tanh: `(exp(2x) - 1) / (exp(2x) + 1)`
    fn tanh_activation(&self) -> Self;

    /// GELU (高速近似): `x * sigmoid(1.702 * x)`
    fn gelu(&self) -> Self;

    /// SiLU (Swish): `x * sigmoid(x)`
    fn silu(&self) -> Self;

    /// Softplus: `ln(1 + exp(x))`
    fn softplus(&self) -> Self;

    /// Mish: `x * tanh(softplus(x))`
    fn mish(&self) -> Self;

    /// ELU: `x if x > 0, else alpha * (exp(x) - 1)`
    fn elu(&self, alpha: Self::Scalar) -> Self;
}

// FloatDType でジェネリックな単一実装
// harp-coreのTensorメソッドに委譲
impl<T: FloatDType, D: Dimension> ActivationExt<D> for Tensor<T, D> {
    type Scalar = T;

    fn relu(&self) -> Self {
        // harp-core: Tensor::relu()
        Tensor::relu(self)
    }

    fn leaky_relu(&self, alpha: T) -> Self {
        // harp-core: Tensor::leaky_relu()
        Tensor::leaky_relu(self, alpha)
    }

    fn sigmoid(&self) -> Self {
        // harp-core: Tensor::sigmoid()
        Tensor::sigmoid(self)
    }

    fn tanh_activation(&self) -> Self {
        // harp-core: Tensor::tanh()
        // 名前の衝突を避けるため tanh_activation として公開
        Tensor::tanh(self)
    }

    fn gelu(&self) -> Self {
        // harp-core: Tensor::gelu()
        Tensor::gelu(self)
    }

    fn silu(&self) -> Self {
        // harp-core: Tensor::silu()
        Tensor::silu(self)
    }

    fn softplus(&self) -> Self {
        // harp-core: Tensor::softplus()
        Tensor::softplus(self)
    }

    fn mish(&self) -> Self {
        // harp-core: Tensor::mish()
        Tensor::mish(self)
    }

    fn elu(&self, alpha: T) -> Self {
        // harp-core: Tensor::elu()
        Tensor::elu(self, alpha)
    }
}

#[cfg(test)]
mod tests {
    use harp::tensor::Dim2;

    use super::*;

    #[test]
    fn test_relu_f32() {
        let input = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let output = input.relu();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_relu_f64() {
        let input = Tensor::<f64, Dim2>::full([2, 3], -1.0);
        let output = input.relu();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_leaky_relu() {
        let input = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let output = input.leaky_relu(0.01);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_sigmoid_f32() {
        let input = Tensor::<f32, Dim2>::zeros([2, 3]);
        let output = input.sigmoid();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_sigmoid_f64() {
        let input = Tensor::<f64, Dim2>::zeros([2, 3]);
        let output = input.sigmoid();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_tanh() {
        let input = Tensor::<f32, Dim2>::zeros([2, 3]);
        let output = input.tanh_activation();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_gelu() {
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = input.gelu();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_silu() {
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = input.silu();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_softplus() {
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = input.softplus();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_mish() {
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = input.mish();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_elu() {
        let input = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let output = input.elu(1.0);
        assert_eq!(output.shape(), &[2, 3]);
    }
}
