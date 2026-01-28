//! Functional API for neural network operations
//!
//! This module provides pure functions for neural network operations.
//! These functions perform the actual computation without managing parameters.
//!
//! # Example
//!
//! ```ignore
//! use eclat_nn::functional;
//! use eclat::tensor::{Tensor, dim::{D2, D4}};
//!
//! // Linear transformation
//! let input: Tensor<D2, f32> = Tensor::input([32, 10]);
//! let weight: Tensor<D2, f32> = Tensor::input([5, 10]);
//! let output = functional::linear(&input, &weight, None);
//!
//! // 2D convolution
//! let input: Tensor<D4, f32> = Tensor::input([1, 3, 32, 32]);
//! let weight: Tensor<D4, f32> = Tensor::input([64, 3, 3, 3]);
//! let output = functional::conv2d(&input, &weight, None, (1, 1), (0, 0), (1, 1));
//! ```

mod activation;
mod attention;
mod conv;
mod linear;

pub use activation::{
    elu, gelu, leaky_relu, log_softmax, prelu, relu, sigmoid, silu, softmax, tanh,
};
pub use attention::{linear_d3, scaled_dot_product_attention};
pub use conv::{
    conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d,
};
pub use linear::linear;
