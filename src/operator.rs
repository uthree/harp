use crate::tensor_node::TensorNode;
use std::fmt::Debug;

pub trait Operator: Debug {
    fn forward(&self, inputs: Vec<TensorNode>) -> Vec<TensorNode>;
}
pub trait Differentiable: Operator {
    fn backward(&self) -> impl Operator;
}
