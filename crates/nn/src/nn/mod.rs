//! Neural Network Module
//!
//! This module provides PyTorch-like neural network abstractions for eclat.
//!
//! # Overview
//!
//! - `Parameter`: Learnable parameter with automatic gradient tracking
//! - `Module`: Base trait for neural network layers
//! - `Linear`: Fully connected layer
//!
//! # Example
//!
//! ```ignore
//! use eclat_nn::nn::{Module, Linear, Parameter};
//! use eclat_nn::optim::{Optimizer, SGD};
//!
//! // Create a linear layer
//! let layer = Linear::new(10, 5, true);
//!
//! // Get all parameters
//! let params = layer.parameters();
//!
//! // Create optimizer
//! let mut optimizer = SGD::new(params, 0.01);
//! ```

mod parameter;
mod module;
mod linear;
mod conv;

pub use parameter::{Parameter, ParameterError};
pub use module::Module;
pub use linear::Linear;
pub use conv::{Conv1d, Conv2d};
