//! 活性化関数層
//!
//! 活性化関数は学習可能なパラメータを持たないステートレスな層です。
//! 活性化関数のメソッドは `functional::activation` モジュールでTensorに追加されています。

use std::collections::HashMap;
use std::marker::PhantomData;

use harp::tensor::{DimDyn, Dimension, FloatDType, Tensor};

use crate::functional::activation::ActivationExt;
use crate::{Module, ParameterMut};

// マクロで活性化層を生成 (FloatDType でジェネリック)
macro_rules! define_activation_layer {
    (
        $(#[$meta:meta])*
        $name:ident, $method:ident
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Default)]
        pub struct $name<T: FloatDType = f32> {
            _dtype: PhantomData<T>,
        }

        impl<T: FloatDType> $name<T> {
            pub fn new() -> Self {
                Self { _dtype: PhantomData }
            }

            pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D> {
                input.$method()
            }
        }

        impl<T: FloatDType> Module<T> for $name<T> {
            fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
                HashMap::new()
            }
            fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
        }
    };
}

// パラメータ付き活性化層のマクロ (FloatDType でジェネリック)
macro_rules! define_activation_layer_with_param {
    (
        $(#[$meta:meta])*
        $name:ident, $method:ident, $param:ident: $param_ty:ty, $default:expr
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub struct $name<T: FloatDType = f32> {
            $param: T,
        }

        impl<T: FloatDType> $name<T> {
            pub fn new($param: T) -> Self {
                Self { $param }
            }

            pub fn $param(&self) -> T {
                self.$param.clone()
            }

            pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D> {
                input.$method(self.$param.clone())
            }
        }

        impl<T: FloatDType> Default for $name<T> {
            fn default() -> Self {
                Self::new(T::from_f64($default))
            }
        }

        impl<T: FloatDType> Module<T> for $name<T> {
            fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
                HashMap::new()
            }
            fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
        }
    };
}

define_activation_layer!(
    /// ReLU 活性化層
    ///
    /// `f(x) = max(0, x)`
    ReLU, relu
);

define_activation_layer!(
    /// Sigmoid 活性化層
    ///
    /// `f(x) = 1 / (1 + exp(-x))`
    Sigmoid, sigmoid
);

define_activation_layer!(
    /// Tanh 活性化層
    ///
    /// `f(x) = (exp(2x) - 1) / (exp(2x) + 1)`
    Tanh, tanh_activation
);

define_activation_layer!(
    /// GELU 活性化層（高速近似）
    ///
    /// `f(x) = x * sigmoid(1.702 * x)`
    GELU, gelu
);

define_activation_layer!(
    /// SiLU (Swish) 活性化層
    ///
    /// `f(x) = x * sigmoid(x)`
    SiLU, silu
);

define_activation_layer!(
    /// Softplus 活性化層
    ///
    /// `f(x) = ln(1 + exp(x))`
    Softplus, softplus
);

define_activation_layer!(
    /// Mish 活性化層
    ///
    /// `f(x) = x * tanh(softplus(x))`
    Mish, mish
);

define_activation_layer_with_param!(
    /// Leaky ReLU 活性化層
    ///
    /// `f(x) = max(alpha * x, x)`
    LeakyReLU, leaky_relu, alpha: f32, 0.01
);

define_activation_layer_with_param!(
    /// ELU 活性化層
    ///
    /// `f(x) = x if x > 0, else alpha * (exp(x) - 1)`
    ELU, elu, alpha: f32, 1.0
);

/// Swish は SiLU の別名
pub type Swish<T = f32> = SiLU<T>;

#[cfg(test)]
mod tests {
    use harp::tensor::Dim2;

    use super::*;

    #[test]
    fn test_relu_layer() {
        let relu = ReLU::<f32>::new();
        let input = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let output = relu.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_relu_layer_f64() {
        let relu = ReLU::<f64>::new();
        let input = Tensor::<f64, Dim2>::full([2, 3], -1.0);
        let output = relu.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_leaky_relu_layer() {
        let leaky_relu = LeakyReLU::<f32>::new(0.01);
        let input = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let output = leaky_relu.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_sigmoid_layer() {
        let sigmoid_layer = Sigmoid::<f32>::new();
        let input = Tensor::<f32, Dim2>::zeros([2, 3]);
        let output = sigmoid_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_sigmoid_layer_f64() {
        let sigmoid_layer = Sigmoid::<f64>::new();
        let input = Tensor::<f64, Dim2>::zeros([2, 3]);
        let output = sigmoid_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_tanh_layer() {
        let tanh_layer = Tanh::<f32>::new();
        let input = Tensor::<f32, Dim2>::zeros([2, 3]);
        let output = tanh_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_gelu_layer() {
        let gelu_layer = GELU::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = gelu_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_silu_layer() {
        let silu_layer = SiLU::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = silu_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_softplus_layer() {
        let softplus_layer = Softplus::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = softplus_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_mish_layer() {
        let mish_layer = Mish::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = mish_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_elu_layer() {
        let elu_layer = ELU::<f32>::new(1.0);
        let input = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let output = elu_layer.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_activation_no_parameters() {
        let mut relu = ReLU::<f32>::new();
        assert_eq!(relu.parameters().len(), 0);
        assert_eq!(relu.num_parameters(), 0);
    }
}
