mod arithmetic;
mod ops;
mod reduce;
mod traits;
pub mod variable;

pub use arithmetic::{Add, Mul, Neg, Recip};
pub use reduce::{Expand, Max, Prod, Sum};
pub use traits::{GradFn, GradNode, GradRoot};
pub use variable::Variable;
