//! Neural Network Module
//!
//! This module provides PyTorch-like neural network abstractions for eclat.
//!
//! # Overview
//!
//! - `Parameter`: Learnable parameter with automatic gradient tracking
//! - `ParameterBase`: Trait for parameters that can be used by optimizers
//! - `Module`: Base trait for neural network layers
//! - `Linear`: Fully connected layer
//! - `Conv1d`, `Conv2d`, `Conv3d`: Convolution layers
//!
//! # Example
//!
//! ```ignore
//! use eclat_nn::nn::{Module, Linear, Parameter, ParameterBase};
//! use eclat_nn::optim::{Optimizer, SGD};
//!
//! // Create a linear layer
//! let layer = Linear::new(10, 5, true);
//!
//! // Get all parameters (as Box<dyn ParameterBase>)
//! let params = layer.parameters();
//!
//! // Create optimizer
//! let mut optimizer = SGD::new(params, 0.01);
//! ```

mod conv;
mod linear;
mod module;
mod parameter;

pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};
pub use linear::Linear;
pub use module::Module;
pub use parameter::{Parameter, ParameterBase, ParameterError};
