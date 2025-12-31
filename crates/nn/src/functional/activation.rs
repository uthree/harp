//! 活性化関数
//!
//! ニューラルネットワークで使用される活性化関数を提供します。
//!
//! 活性化関数は `harp::tensor::Tensor` に直接実装されています。
//! このモジュールは互換性のためにエイリアス（`tanh_activation`）を提供します。
//!
//! # 使い方
//!
//! ```ignore
//! use harp::tensor::{Tensor, Dim2};
//!
//! let input = Tensor::<f32, Dim2>::ones([2, 3]);
//! let output = input.relu();      // harp-core の Tensor メソッド
//! let output = input.sigmoid();   // harp-core の Tensor メソッド
//! let output = input.tanh();      // harp-core の Tensor メソッド
//! ```
//!
//! # 関数一覧 (harp-core で提供)
//!
//! - `relu()` - ReLU: `max(0, x)`
//! - `leaky_relu(alpha)` - Leaky ReLU: `max(alpha * x, x)`
//! - `sigmoid()` - Sigmoid: `1 / (1 + exp(-x))`
//! - `tanh()` - Tanh: `(exp(2x) - 1) / (exp(2x) + 1)`
//! - `gelu()` - GELU (高速近似): `x * sigmoid(1.702 * x)`
//! - `silu()` - SiLU (Swish): `x * sigmoid(x)`
//! - `softplus()` - Softplus: `ln(1 + exp(x))`
//! - `mish()` - Mish: `x * tanh(softplus(x))`
//! - `elu(alpha)` - ELU: `x if x > 0, else alpha * (exp(x) - 1)`

#[cfg(test)]
mod tests {
    use harp::tensor::{Dim2, FloatDType, Tensor};

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
        let output = input.tanh();
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
