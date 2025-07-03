use crate::shape::tracker::ShapeTracker;
use crate::tensor_node::{TensorNode, TensorNodeStore};
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct Graph {
    variables: Vec<String>, //variables for shapetracker
    inputs: Vec<TensorNode>,
    outputs: Vec<TensorNode>,
    nodes: Vec<Arc<RefCell<TensorNodeStore>>>,
}

impl Graph {
    fn new() -> Self {
        Self::default()
    }

    fn input(&mut self) -> TensorNode {
        todo!()
        //let tensor_node_store = TensorNodeStore::new(ShapeTracker)
    }
}
