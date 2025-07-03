use crate::operator::Operator;
use crate::tensor_node::{TensorNode, TensorNodeStore};
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Graph {
    pub variables: Vec<String>, //variables for shapetracker
    pub inputs: Vec<TensorNode>,
    pub outputs: Vec<TensorNode>,
    pub nodes: Vec<Arc<RefCell<TensorNodeStore>>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            variables: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub fn input(&mut self) -> TensorNode {
        let node = Arc::new(RefCell::new(TensorNodeStore::new(
            Operator::Input,
            vec![],
            crate::shape::tracker::ShapeTracker::full(vec![]), // 仮のShapeTracker
            crate::tensor_node::DataType::Float32, // 仮のDataType
        )));
        self.nodes.push(node.clone());
        Arc::downgrade(&node)
    }
}
