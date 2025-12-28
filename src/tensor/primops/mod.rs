//! Primitive operations (primops) for tensors
//!
//! This module contains the minimal set of operations from which all other
//! tensor operations can be composed. Following tinygrad's design philosophy.
//!
//! ## Categories
//!
//! - **Initialization**: Const, Rand
//! - **Binary**: Add, Mul, Max, Idiv, Rem
//! - **Unary**: Neg, Recip, Sqrt, Log2, Exp2, Sin
//! - **Bitwise**: BitAnd, BitOr, BitXor, BitNot, Shl, Shr
//! - **Reduce**: Reduce(Add), Reduce(Mul), Reduce(Max)
//! - **Movement**: Squeeze, Unsqueeze, Repeat, Reshape, Contiguous
//!
//! ## Gradient Functions
//!
//! Each primop has a corresponding GradFn implementation for automatic differentiation.
//! Fused operations also have gradient support via symbolic differentiation.

mod binary;
mod bitwise;
mod grad;
mod init;
mod movement;
mod reduce;
mod unary;

// Re-export unary operation traits
pub use unary::{Exp2, Floor, Log2, Recip, Sin, Sqrt};

// Re-export unary gradient functions
pub use unary::{
    CastF32ToF64Backward, CastF64ToF32Backward, Exp2Backward, Log2Backward, NegBackward,
    RecipBackward, SinBackward, SqrtBackward,
};

// Re-export binary gradient functions
pub use binary::{AddBackward, MaxBackward, MulBackward};

// Reduce gradient functions are pub(crate) - used internally only

// Re-export general gradient utilities and fused operation gradients
pub use grad::{CloneBackward, FusedElementwiseBackward, FusedElementwiseReduceBackward};
