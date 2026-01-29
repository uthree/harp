//! OpenCL kernel operations.

pub mod compare;
pub mod elementwise;
pub mod movement;
pub mod reduce;

pub use compare::*;
pub use elementwise::*;
pub use movement::*;
pub use reduce::*;
