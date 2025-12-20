mod arithmetic;
mod ops;
mod reduce;
mod traits;
pub mod variable;

pub use arithmetic::{Add, Mul, Neg, Recip};
pub use reduce::{Expand, Sum};
pub use traits::{GradFn, GradNode, GradRoot, GradientInto};
pub use variable::Variable;
