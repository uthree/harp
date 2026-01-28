//! Neural Network Module for Eclat
//!
//! This crate provides PyTorch-like neural network modules and optimizers.
//!
//! # Overview
//!
//! - `layers`: Neural network layers and parameter management
//! - `functional`: Pure functions for neural network operations
//! - `optim`: Optimization algorithms (SGD, Adam)
//!
//! # Example
//!
//! ```ignore
//! use eclat_nn::layers::{Module, Linear, Parameter, ParameterBase};
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

pub mod functional;
pub mod layers;
pub mod optim;

// Re-export commonly used types from layers
pub use layers::{
    Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Linear, Module,
    MultiheadAttention, PReLU, Parameter, ParameterBase, ParameterError,
};
pub use optim::{Adam, OptimError, Optimizer, SGD};
