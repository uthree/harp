use crate::tensor_node::{TensorNode, TensorNodeStore};
use std::cell::RefCell;
use std::sync::{Arc, Weak};
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<Arc<RefCell<TensorNodeStore>>>,
}
