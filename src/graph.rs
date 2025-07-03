use crate::operator::Operator;
use crate::prelude::*;
use crate::shape::symbolic::Expr;
use crate::tensor_node::{DataType, TensorNode, TensorNodeStore};
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

    pub fn input(&mut self, shape: Vec<Expr>, dtype: DataType) -> TensorNode {
        let node = Arc::new(RefCell::new(TensorNodeStore::new(
            Operator::Input,
            vec![],
            ShapeTracker::full(shape),
            dtype,
        )));
        self.nodes.push(node.clone());
        Arc::downgrade(&node)
    }
}
