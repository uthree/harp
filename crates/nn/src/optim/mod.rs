//! オプティマイザ
//!
//! パラメータ更新のための最適化アルゴリズムを提供します。

mod momentum;
mod sgd;

pub use momentum::Momentum;
pub use sgd::SGD;

use harp::tensor::FloatDType;

use crate::Module;

/// オプティマイザの基底トレイト
///
/// # Type Parameters
///
/// * `T` - パラメータのデータ型
pub trait Optimizer<T: FloatDType> {
    /// パラメータを更新（勾配降下ステップを実行）
    fn step<M: Module<T>>(&mut self, module: &mut M);
}
