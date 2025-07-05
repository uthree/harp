pub mod graph;
pub mod macros;
pub mod ops;
pub mod shape;
pub mod tensor;

pub mod prelude {
    pub use crate::graph::{Graph, GraphRef};
    pub use crate::ops::Operator;
    pub use crate::shape::tracker::ShapeTracker;
    pub use crate::tensor::{Tensor, TensorRef};
}