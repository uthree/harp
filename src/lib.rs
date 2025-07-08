pub mod graph;
pub mod macros;
pub mod shape;

pub mod prelude {
    pub use crate::graph::graph::Graph;
    pub use crate::graph::operator::Operator;
    pub use crate::shape::tracker::ShapeTracker;
    pub use crate::graph::tensor::Tensor;
}

