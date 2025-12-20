mod arithmetic;
mod ops;
mod traits;
pub mod variable;

pub use arithmetic::{Add, Mul, Neg, Recip};
pub use traits::{GradFn, GradNode, GradRoot, GradientInto};
pub use variable::Variable;
