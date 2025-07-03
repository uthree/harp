use crate::operator::Operator;
use crate::prelude::*;
use crate::shape::symbolic::Expr;
use crate::tensor_node::{DataType, TensorNode};
use crate::unique_id;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GraphStore {
    pub id: usize,
    pub variables: Vec<String>, //variables for shapetracker
    pub inputs: Vec<TensorNode>,
    pub outputs: Vec<TensorNode>,
    pub nodes: Vec<TensorNode>,
}

#[derive(Clone, PartialEq)]
pub struct Graph {
    store: Arc<RefCell<GraphStore>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            store: Arc::new(RefCell::new(GraphStore {
                id: unique_id::next_id(),
                variables: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                nodes: Vec::new(),
            })),
        }
    }

    pub fn input(&mut self, shape: Vec<Expr>, dtype: DataType) -> TensorNode {
        let node = TensorNode::new(
            Operator::Input,
            vec![],
            ShapeTracker::full(shape),
            dtype,
            self.clone(),
        );
        self.store.borrow_mut().nodes.push(node.clone());
        node
    }

    pub(crate) fn apply_node(&mut self, node: TensorNode) {
        self.store.borrow_mut().nodes.push(node);
    }
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph {{ id={}, with {} nodes }}",
            self.store.borrow().id,
            self.store.borrow().nodes.len()
        )
    }
}
