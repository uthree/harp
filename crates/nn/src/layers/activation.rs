//! 活性化関数層
//!
//! 活性化関数は学習可能なパラメータを持たないステートレスな層です。
//! 関数形式と層形式の両方を提供します。

use std::collections::HashMap;
use std::marker::PhantomData;

use harp::tensor::{DimDyn, Dimension, FloatDType, Tensor};

use crate::{Module, ParameterMut};

// ============================================================================
// 活性化操作トレイト（sealed）
// ============================================================================

mod sealed {
    use harp::tensor::{Dimension, Recip, Tensor};

    /// 活性化関数操作を提供するトレイト
    pub trait ActivationOps<D: Dimension>: Sized {
        fn relu(&self) -> Self;
        fn leaky_relu(&self, alpha: Self::Scalar) -> Self;
        fn sigmoid(&self) -> Self;
        fn tanh_act(&self) -> Self;
        fn gelu(&self) -> Self;
        fn silu(&self) -> Self;
        fn softplus(&self) -> Self;
        fn mish(&self) -> Self;
        fn elu(&self, alpha: Self::Scalar) -> Self;

        type Scalar;
    }

    impl<D: Dimension> ActivationOps<D> for Tensor<f32, D> {
        type Scalar = f32;

        fn relu(&self) -> Self {
            self.maximum_scalar(0.0)
        }
        fn leaky_relu(&self, alpha: f32) -> Self {
            let scaled = self * alpha;
            self.maximum(&scaled)
        }
        fn sigmoid(&self) -> Self {
            let neg_x = -self;
            let exp_neg_x = neg_x.exp();
            let one_plus = exp_neg_x + 1.0;
            one_plus.recip()
        }
        fn tanh_act(&self) -> Self {
            let two_x = self * 2.0;
            let exp_2x = two_x.exp();
            let numerator = &exp_2x - 1.0;
            let denominator = &exp_2x + 1.0;
            numerator / denominator
        }
        fn gelu(&self) -> Self {
            let scaled = self * 1.702;
            let sig = scaled.sigmoid();
            self * &sig
        }
        fn silu(&self) -> Self {
            let sig = self.sigmoid();
            self * &sig
        }
        fn softplus(&self) -> Self {
            let exp_x = self.exp();
            let one_plus = exp_x + 1.0;
            one_plus.ln()
        }
        fn mish(&self) -> Self {
            let sp = self.softplus();
            let tanh_sp = sp.tanh();
            self * &tanh_sp
        }
        fn elu(&self, alpha: f32) -> Self {
            let exp_x = self.exp();
            let exp_minus_1 = exp_x - 1.0;
            let negative_part = exp_minus_1 * alpha;
            let positive = self.relu();
            positive + (-(&negative_part - &negative_part.relu()))
        }
    }

    impl<D: Dimension> ActivationOps<D> for Tensor<f64, D> {
        type Scalar = f64;

        fn relu(&self) -> Self {
            self.maximum_scalar(0.0)
        }
        fn leaky_relu(&self, alpha: f64) -> Self {
            let scaled = self * alpha;
            self.maximum(&scaled)
        }
        fn sigmoid(&self) -> Self {
            let neg_x = -self;
            let exp_neg_x = neg_x.exp();
            let one_plus = exp_neg_x + 1.0;
            one_plus.recip()
        }
        fn tanh_act(&self) -> Self {
            let two_x = self * 2.0;
            let exp_2x = two_x.exp();
            let numerator = &exp_2x - 1.0;
            let denominator = &exp_2x + 1.0;
            numerator / denominator
        }
        fn gelu(&self) -> Self {
            let scaled = self * 1.702;
            let sig = scaled.sigmoid();
            self * &sig
        }
        fn silu(&self) -> Self {
            let sig = self.sigmoid();
            self * &sig
        }
        fn softplus(&self) -> Self {
            let exp_x = self.exp();
            let one_plus = exp_x + 1.0;
            one_plus.ln()
        }
        fn mish(&self) -> Self {
            let sp = self.softplus();
            let tanh_sp = sp.tanh();
            self * &tanh_sp
        }
        fn elu(&self, alpha: f64) -> Self {
            let exp_x = self.exp();
            let exp_minus_1 = exp_x - 1.0;
            let negative_part = exp_minus_1 * alpha;
            let positive = self.relu();
            positive + (-(&negative_part - &negative_part.relu()))
        }
    }
}

use sealed::ActivationOps;

// ============================================================================
// 活性化関数層（パラメータなし）
// ============================================================================

/// ReLU 活性化層
///
/// `f(x) = max(0, x)`
///
/// # Example
///
/// ```ignore
/// let relu = ReLU::<f32>::new();
/// let output = relu.forward(&input);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ReLU<T: FloatDType = f32> {
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> ReLU<T> {
    pub fn new() -> Self {
        Self {
            _dtype: PhantomData,
        }
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D>,
    {
        input.relu()
    }
}

impl<T: FloatDType> Module<T> for ReLU<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// Leaky ReLU 活性化層
///
/// `f(x) = max(alpha * x, x)`
///
/// # Example
///
/// ```ignore
/// let leaky_relu = LeakyReLU::<f32>::new(0.01);
/// let output = leaky_relu.forward(&input);
/// ```
#[derive(Debug, Clone)]
pub struct LeakyReLU<T: FloatDType = f32> {
    alpha: T,
}

impl<T: FloatDType> LeakyReLU<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }

    /// alpha 値を取得
    pub fn alpha(&self) -> T {
        self.alpha.clone()
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D, Scalar = T>,
    {
        input.leaky_relu(self.alpha.clone())
    }
}

impl Default for LeakyReLU<f32> {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Default for LeakyReLU<f64> {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl<T: FloatDType> Module<T> for LeakyReLU<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// Sigmoid 活性化層
///
/// `f(x) = 1 / (1 + exp(-x))`
///
/// # Example
///
/// ```ignore
/// let sigmoid = Sigmoid::<f32>::new();
/// let output = sigmoid.forward(&input);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Sigmoid<T: FloatDType = f32> {
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Sigmoid<T> {
    pub fn new() -> Self {
        Self {
            _dtype: PhantomData,
        }
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D>,
    {
        input.sigmoid()
    }
}

impl<T: FloatDType> Module<T> for Sigmoid<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// Tanh 活性化層
///
/// `f(x) = (exp(2x) - 1) / (exp(2x) + 1)`
///
/// # Example
///
/// ```ignore
/// let tanh = Tanh::<f32>::new();
/// let output = tanh.forward(&input);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Tanh<T: FloatDType = f32> {
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Tanh<T> {
    pub fn new() -> Self {
        Self {
            _dtype: PhantomData,
        }
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D>,
    {
        input.tanh_act()
    }
}

impl<T: FloatDType> Module<T> for Tanh<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// GELU 活性化層（高速近似）
///
/// `f(x) = x * sigmoid(1.702 * x)`
///
/// # Example
///
/// ```ignore
/// let gelu = GELU::<f32>::new();
/// let output = gelu.forward(&input);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GELU<T: FloatDType = f32> {
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> GELU<T> {
    pub fn new() -> Self {
        Self {
            _dtype: PhantomData,
        }
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D>,
    {
        input.gelu()
    }
}

impl<T: FloatDType> Module<T> for GELU<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// SiLU (Swish) 活性化層
///
/// `f(x) = x * sigmoid(x)`
///
/// # Example
///
/// ```ignore
/// let silu = SiLU::<f32>::new();
/// let output = silu.forward(&input);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SiLU<T: FloatDType = f32> {
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> SiLU<T> {
    pub fn new() -> Self {
        Self {
            _dtype: PhantomData,
        }
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D>,
    {
        input.silu()
    }
}

impl<T: FloatDType> Module<T> for SiLU<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// Softplus 活性化層
///
/// `f(x) = ln(1 + exp(x))`
///
/// # Example
///
/// ```ignore
/// let softplus = Softplus::<f32>::new();
/// let output = softplus.forward(&input);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Softplus<T: FloatDType = f32> {
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Softplus<T> {
    pub fn new() -> Self {
        Self {
            _dtype: PhantomData,
        }
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D>,
    {
        input.softplus()
    }
}

impl<T: FloatDType> Module<T> for Softplus<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// Mish 活性化層
///
/// `f(x) = x * tanh(softplus(x))`
///
/// # Example
///
/// ```ignore
/// let mish = Mish::<f32>::new();
/// let output = mish.forward(&input);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Mish<T: FloatDType = f32> {
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Mish<T> {
    pub fn new() -> Self {
        Self {
            _dtype: PhantomData,
        }
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D>,
    {
        input.mish()
    }
}

impl<T: FloatDType> Module<T> for Mish<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

/// ELU 活性化層
///
/// `f(x) = x if x > 0, else alpha * (exp(x) - 1)`
///
/// # Example
///
/// ```ignore
/// let elu = ELU::<f32>::new(1.0);
/// let output = elu.forward(&input);
/// ```
#[derive(Debug, Clone)]
pub struct ELU<T: FloatDType = f32> {
    alpha: T,
}

impl<T: FloatDType> ELU<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }

    /// alpha 値を取得
    pub fn alpha(&self) -> T {
        self.alpha.clone()
    }

    /// 順伝播
    pub fn forward<D: Dimension>(&self, input: &Tensor<T, D>) -> Tensor<T, D>
    where
        Tensor<T, D>: ActivationOps<D, Scalar = T>,
    {
        input.elu(self.alpha.clone())
    }
}

impl Default for ELU<f32> {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Default for ELU<f64> {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<T: FloatDType> Module<T> for ELU<T> {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<T>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<T, DimDyn>>) {}
}

// ============================================================================
// 便利なエイリアス
// ============================================================================

/// Swish は SiLU の別名
pub type Swish<T = f32> = SiLU<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use harp::tensor::Dim2;

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
        let sigmoid = Sigmoid::<f32>::new();
        let input = Tensor::<f32, Dim2>::zeros([2, 3]);
        let output = sigmoid.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_sigmoid_layer_f64() {
        let sigmoid = Sigmoid::<f64>::new();
        let input = Tensor::<f64, Dim2>::zeros([2, 3]);
        let output = sigmoid.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_tanh_layer() {
        let tanh = Tanh::<f32>::new();
        let input = Tensor::<f32, Dim2>::zeros([2, 3]);
        let output = tanh.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_gelu_layer() {
        let gelu = GELU::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = gelu.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_silu_layer() {
        let silu = SiLU::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = silu.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_softplus_layer() {
        let softplus = Softplus::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = softplus.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_mish_layer() {
        let mish = Mish::<f32>::new();
        let input = Tensor::<f32, Dim2>::ones([2, 3]);
        let output = mish.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_elu_layer() {
        let elu = ELU::<f32>::new(1.0);
        let input = Tensor::<f32, Dim2>::full([2, 3], -1.0);
        let output = elu.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_activation_no_parameters() {
        let mut relu = ReLU::<f32>::new();
        assert_eq!(relu.parameters().len(), 0);
        assert_eq!(relu.num_parameters(), 0);
    }
}
