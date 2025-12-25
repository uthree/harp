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

mod binary;
mod grad;
mod init;
mod movement;
mod reduce;
mod unary;

// Helper functions and gradient structs are used internally by the primops submodules
