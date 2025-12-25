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

// Re-export gradient functions for use in backward pass
pub use grad::{
    // Basic gradients
    AddBackward,
    CloneBackward,
    Exp2Backward,
    // Fused operation gradients
    FusedElementwiseBackward,
    FusedElementwiseReduceBackward,
    Log2Backward,
    MaxBackward,
    MulBackward,
    NegBackward,
    RecipBackward,
    ReduceMaxBackward,
    ReduceMulBackward,
    // Reduce gradients
    ReduceSumBackward,
    SinBackward,
    SqrtBackward,
};
