//! Harp Neural Network Module
//!
//! ニューラルネットワーク構築のための層・損失関数・オプティマイザを提供します。
//!
//! # 基本概念
//!
//! - **Module**: 学習可能なパラメータを持つ計算ユニットの基底トレイト
//! - **Layer**: Linear, Conv2d などの基本的なレイヤー
//! - **Loss**: MSE, CrossEntropy などの損失関数
//! - **Optimizer**: SGD, Adam などの最適化アルゴリズム
//!
//! # Example
//!
//! ```ignore
//! use harp_nn::prelude::*;
//!
//! let linear = Linear::new(784, 128);
//! let output = linear.forward(&input);
//! ```

pub mod functional;
pub mod layers;
pub mod loss;
pub mod module;
pub mod optim;

// Derive マクロを re-export
pub use harp_nn_derive::Module;

pub mod prelude {
    //! 一般的に使用される型のre-export
    pub use crate::functional;
    pub use crate::layers::*;
    pub use crate::loss::*;
    pub use crate::module::*;
    pub use crate::optim::*;
    pub use harp_nn_derive::Module;
}

pub use layers::*;
pub use loss::*;
pub use module::*;
pub use optim::*;
