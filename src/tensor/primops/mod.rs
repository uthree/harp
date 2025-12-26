//! Primitive operations (primops) for tensors
//!
//! This module contains the minimal set of operations from which all other
//! tensor operations can be composed. Following tinygrad's design philosophy.
//!
//! ## Categories
//!
//! - **Initialization**: Const, Rand
//! - **Binary**: Add, Mul, Max, Idiv
//! - **Unary**: Neg, Recip, Sqrt, Log2, Exp2, Sin
//! - **Reduce**: Reduce(Add), Reduce(Mul), Reduce(Max)
//! - **Movement**: Squeeze, Unsqueeze, Repeat, Reshape, Contiguous
//!
//! ## Gradient Functions
//!
//! Each primop has a corresponding GradFn implementation for automatic differentiation.
//! Fused operations also have gradient support via symbolic differentiation.

mod binary;
mod grad;
mod init;
mod movement;
mod reduce;
mod unary;

// Re-export unary operation traits
pub use unary::{Exp2, Floor, Log2, Recip, Sin, Sqrt};

// Re-export unary gradient functions
pub use unary::{
    Exp2Backward, Log2Backward, NegBackward, RecipBackward, SinBackward, SqrtBackward,
};

// Re-export binary gradient functions
pub use binary::{AddBackward, MaxBackward, MulBackward};

// Re-export reduce gradient functions
pub use reduce::{ReduceMaxBackward, ReduceMulBackward, ReduceSumBackward};

// Re-export general gradient utilities and fused operation gradients
pub use grad::{CloneBackward, FusedElementwiseBackward, FusedElementwiseReduceBackward};
