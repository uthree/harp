mod grad_fns;
mod ops;
mod traits;
pub mod variable;

pub use grad_fns::{Add, Mul, Neg, Recip};
pub use traits::{GradFn, GradNode, GradRoot};
pub use variable::Variable;
