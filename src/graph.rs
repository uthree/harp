use crate::tensor_node::{TensorNode, TensorNodeStore};
use std::cell::RefCell;
use std::sync::Arc;
#[derive(Debug, Clone)]
pub struct Graph {
    variables: Vec<String>,
    inputs: Vec<TensorNode>,
    outputs: Vec<TensorNode>,
    nodes: Vec<Arc<RefCell<TensorNodeStore>>>,
}

impl Graph {}
