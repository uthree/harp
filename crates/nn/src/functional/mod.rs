//! Functional operations for neural networks
//!
//! このモジュールはニューラルネットワークの操作を関数として提供します。
//! レイヤー（Module）を使わずに、直接テンソルに対して操作を適用できます。
//!
//! # Example
//!
//! ```ignore
//! use harp_nn::functional::{max_pool2d, avg_pool2d};
//!
//! let input: Tensor<f32, Dim4> = ...;
//! let output = max_pool2d(&input, (2, 2), (2, 2), (0, 0));
//! ```

mod pooling;

pub use pooling::*;
