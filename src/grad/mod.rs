//! Automatic differentiation module
//!
//! This module provides reverse-mode automatic differentiation (backpropagation)
//! for computation graphs.
//!
//! # Overview
//!
//! The gradient computation follows the source transformation approach:
//! - `backward()` builds a new computation graph representing the gradients
//! - The gradient graph can be optimized and executed like any other graph
//! - This enables fusion and optimization of backward passes
//!
//! # Example
//!
//! ```ignore
//! use eclat::grad::{backward, Differentiable};
//! use eclat::graph::{input, Expr};
//! use eclat::ast::DType;
//!
//! // Forward computation
//! let x = input(vec![Expr::Const(10)], DType::F32);
//! let y = (&x * &x).sum(0);  // y = sum(x^2)
//!
//! // Compute gradient (lazy - returns a GraphNode)
//! let grads = backward(&y, &[&x]);
//! let dx = grads.get(&x).unwrap();  // dx = 2*x
//!
//! // Or use the extension trait
//! let dx = y.grad(&x).unwrap();
//! ```
//!
//! # Supported Operations
//!
//! Gradients are implemented for:
//!
//! ## Elementwise Operations
//! - `add(a, b)`: `∂a = ∂out`, `∂b = ∂out`
//! - `mul(a, b)`: `∂a = ∂out * b`, `∂b = ∂out * a`
//! - `neg(a)`: `∂a = -∂out`
//! - `recip(a)`: `∂a = -∂out / a²`
//! - `sqrt(a)`: `∂a = ∂out / (2 * sqrt(a))`
//! - `exp(a)`: `∂a = ∂out * exp(a)`
//! - `log(a)`: `∂a = ∂out / a`
//! - `sin(a)`: `∂a = ∂out * cos(a)`
//! - `cos(a)`: `∂a = -∂out * sin(a)`
//!
//! ## Reduce Operations
//! - `sum(a, axis)`: `∂a = expand(∂out, axis)`
//! - `max(a, axis)`: `∂a = one_hot_mask * expand(∂out, axis)`
//! - `min(a, axis)`: `∂a = one_hot_mask * expand(∂out, axis)`
//! - `prod(a, axis)`: `∂a = ∂out * prod / a`
//!
//! ## View Operations
//! - `reshape(a)`: `∂a = reshape(∂out, original_shape)`
//! - `permute(a)`: `∂a = permute(∂out, inverse_axes)`
//! - `expand(a)`: `∂a = sum(∂out, axis)`
//! - `squeeze(a)`: `∂a = unsqueeze(∂out, axis)`
//! - `unsqueeze(a)`: `∂a = squeeze(∂out, axis)`

mod backward;
mod context;
mod rules;

// Re-export main API
pub use backward::{Differentiable, backward, grad, grads};
pub use context::{GradContext, GradResult};
pub use rules::{VjpResult, compute_vjp};
