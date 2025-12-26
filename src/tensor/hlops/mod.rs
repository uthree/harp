//! High-level operations (hlops) for tensors
//!
//! These operations are composed from primitive operations (primops).
//! They provide convenient APIs but are internally implemented using primops.
//!
//! ## Arithmetic
//! - Sub(a, b) = Add(a, Neg(b))
//! - Div(a, b) = Mul(a, Recip(b))
//!
//! ## Transcendental
//! - Exp(x) = Exp2(x * log2(e))
//! - Ln(x) = Log2(x) * ln(2)
//! - Cos(x) = Sin(x + π/2)
//!
//! ## Activation
//! - ReLU(x) = Max(x, 0)
//! - Sigmoid(x) = 1 / (1 + Exp(-x))
//! - Tanh(x) = (Exp(2x) - 1) / (Exp(2x) + 1)
//!
//! ## Linear Algebra
//! - MatMul = Unsqueeze + Mul + ReduceSum

mod activation;
mod arithmetic;
mod linalg;
mod transcendental;

// hlops modules implement methods on Tensor directly,
// no re-exports needed
