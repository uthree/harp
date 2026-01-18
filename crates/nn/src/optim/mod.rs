//! Optimizers Module
//!
//! This module provides optimization algorithms for training neural networks.
//!
//! # Overview
//!
//! - `Optimizer`: Base trait for all optimizers
//! - `SGD`: Stochastic Gradient Descent with optional momentum
//! - `Adam`: Adaptive Moment Estimation optimizer
//!
//! # Example
//!
//! ```ignore
//! use eclat_nn::nn::{Module, Linear};
//! use eclat_nn::optim::{Optimizer, SGD, Adam};
//!
//! let layer = Linear::new(10, 5, true);
//! let params = layer.parameters();
//!
//! // Create SGD optimizer
//! let mut sgd = SGD::new(params.clone(), 0.01);
//!
//! // Or Adam optimizer
//! let mut adam = Adam::new(params, 0.001);
//!
//! // Training loop
//! optimizer.zero_grad();
//! // ... forward pass, compute loss, backward pass ...
//! optimizer.step().unwrap();
//! ```

mod adam;
mod optimizer;
mod sgd;

pub use adam::Adam;
pub use optimizer::{OptimError, Optimizer, ParamData, get_param_data};
pub use sgd::SGD;
