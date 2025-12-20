mod arithmetic;
mod ops;
mod reduce;
mod traits;
pub mod variable;

pub use arithmetic::{Add, Mul, Neg, Recip};
pub use reduce::{Expand, Sum};
pub use traits::{GradFn, GradInto, GradNode, GradRoot};
pub use variable::Variable;
