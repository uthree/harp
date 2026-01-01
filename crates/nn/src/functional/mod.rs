//! Functional operations for neural networks
//!
//! このモジュールはニューラルネットワークの操作をテンソルのメソッドとして提供します。
//! レイヤー（Module）を使わずに、直接テンソルに対して操作を適用できます。
//!
//! # 活性化関数
//!
//! 活性化関数は `harp::tensor::Tensor` に直接実装されています。
//!
//! ```ignore
//! use harp::tensor::{Tensor, Dim2};
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
pub mod interpolate;
pub mod pooling;

// interpolate functions re-export
pub use interpolate::{bilinear2d, linear1d, nearest1d, nearest2d, nearest3d, trilinear3d};

// pooling functions re-export
pub use pooling::{
    adaptive_avg_pool2d, adaptive_max_pool2d, avg_pool1d, avg_pool2d, avg_pool3d, max_pool1d,
    max_pool2d, max_pool3d,
};
