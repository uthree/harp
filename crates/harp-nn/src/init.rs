//! パラメータ初期化関数
//!
//! ニューラルネットワークのパラメータを初期化するための関数群。
//!
//! # 注意
//!
//! 現在、乱数初期化（Xavier, He等）は未実装です。
//! 定数初期化のみサポートしています。

use harp::autograd::Tensor;

/// 定数で初期化
///
/// # Arguments
///
/// * `shape` - テンソルの形状
/// * `value` - 初期化する値
///
/// # Examples
///
/// ```
/// use harp_nn::init;
///
/// let tensor = init::constant(vec![10, 20], 0.5);
/// ```
pub fn constant(shape: Vec<usize>, value: f32) -> Tensor {
    let zeros = Tensor::zeros(shape);
    &zeros + value
}

// TODO: 将来的に実装する乱数初期化関数
//
// pub fn xavier_uniform(shape: Vec<usize>) -> Tensor
// pub fn xavier_normal(shape: Vec<usize>) -> Tensor
// pub fn kaiming_uniform(shape: Vec<usize>) -> Tensor
// pub fn kaiming_normal(shape: Vec<usize>) -> Tensor
// pub fn uniform(shape: Vec<usize>, low: f32, high: f32) -> Tensor
// pub fn normal(shape: Vec<usize>, mean: f32, std: f32) -> Tensor

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let tensor = constant(vec![10, 20], 2.5);
        let shape = tensor.data.view.shape();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_constant_zero() {
        let tensor = constant(vec![5, 5], 0.0);
        let shape = tensor.data.view.shape();
        assert_eq!(shape.len(), 2);
    }
}
