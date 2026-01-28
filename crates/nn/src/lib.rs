//! Neural Network Module for Eclat
//!
//! This crate provides PyTorch-like neural network modules and optimizers.
//!
//! # Overview
//!
//! - `nn`: Neural network layers and parameter management
//! - `optim`: Optimization algorithms (SGD, Adam)
//!
//! # Example
//!
//! ```ignore
//! use eclat_nn::nn::{Module, Linear, Parameter, ParameterBase};
//! use eclat_nn::optim::{Optimizer, SGD, Adam};
//!
//! // Create a layer
//! let layer = Linear::new(10, 5, true);
//!
//! // Get parameters (as Box<dyn ParameterBase>)
//! let params = layer.parameters();
//!
//! // Create optimizer
//! let mut optimizer = Adam::new(params, 0.001);
//!
//! // Training loop
//! optimizer.zero_grad();
//! // ... forward, backward ...
//! optimizer.step().unwrap();
//! ```

pub mod nn;
pub mod optim;

// Re-export commonly used types
pub use nn::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Linear, Module, Parameter, ParameterBase, ParameterError};
pub use optim::{Adam, OptimError, Optimizer, SGD};
