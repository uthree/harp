pub mod dtype;
pub mod graph;
pub mod interpreter;
pub mod macros;
pub mod node;
pub mod operator;
pub mod optimizer;
pub mod shape;
pub mod tensor;

pub mod prelude {
    pub use crate::graph::Graph;
    pub use crate::operator::Operator;
    pub use crate::shape::tracker::ShapeTracker;
    pub use crate::tensor::Tensor;
}
