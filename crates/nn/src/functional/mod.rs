//! Functional operations for neural networks
//!
//! このモジュールはニューラルネットワークの操作をテンソルのメソッドとして提供します。
//! レイヤー（Module）を使わずに、直接テンソルに対して操作を適用できます。
//!
//! # 活性化関数
//!
//! `activation`モジュールをインポートすると、テンソルに活性化関数メソッドが追加されます。
//!
//! ```ignore
//! use harp_nn::functional::activation::ActivationExt;
//!
//! let input = Tensor::<f32, Dim2>::ones([2, 3]);
//! let output = input.relu();
//! let activated = output.sigmoid();
//! ```
//!
//! # プーリング
//!
//! プーリング関数は直接呼び出せます。
//!
//! ```ignore
//! use harp_nn::functional::pooling::{max_pool2d, avg_pool2d};
//!
//! let input: Tensor<f32, Dim4> = ...;
//! let output = max_pool2d(&input, (2, 2), (2, 2), (0, 0));
//! ```

pub mod activation;
pub mod pooling;

// pooling functions re-export
pub use pooling::{
    adaptive_avg_pool2d, adaptive_max_pool2d, avg_pool1d, avg_pool2d, avg_pool3d,
    global_avg_pool1d, global_avg_pool2d, global_avg_pool3d, global_max_pool1d, global_max_pool2d,
    global_max_pool3d, max_pool1d, max_pool2d, max_pool3d,
};
