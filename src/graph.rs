use crate::ops::Input;
use crate::shape::symbolic::Expr;
use crate::{prelude::*, tensor::TensorData};
use std::{cell::RefCell, sync::Arc, sync::Weak};

pub struct GraphData {
    pub input_nodes: Vec<Tensor>,
    pub output_nodes: Vec<Tensor>,
}

pub struct Graph {
    pub data: Arc<RefCell<GraphData>>,
}

#[derive(Debug)]
pub struct GraphRef {
    pub data: Weak<RefCell<GraphData>>,
}

impl PartialEq for GraphRef {
    fn eq(&self, other: &Self) -> bool {
        self.clone()
            .upgrade()
            .and_then(|s| {
                other
                    .clone()
                    .upgrade()
                    .map(|o| Arc::ptr_eq(&s.data, &o.data))
            })
            .unwrap_or(false)
    }
}

impl Eq for GraphRef {}

impl Graph {
    pub fn new() -> Self {
        let content = GraphData {
            input_nodes: vec![],
            output_nodes: vec![],
        };
        let data = Arc::new(RefCell::new(content));
        Graph { data }
    }

    pub fn input(&mut self, shape: Vec<Expr>) -> Tensor {
        let shape_tracker = ShapeTracker::full(self.clone().downgrade(), shape);
        let tensor_data = TensorData {
            graph: self.clone().downgrade(),
            shape_tracker: shape_tracker,
            inputs: vec![],
            operator: Box::new(Input {}),
        };
        let tensor = Tensor {
            data: Arc::new(RefCell::new(tensor_data)),
        };
        let mut data = self.data.borrow_mut();
        data.input_nodes.push(tensor.clone());
        tensor
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Graph {
            data: self.data.clone(),
        }
    }
}

impl Graph {
    pub fn downgrade(self) -> GraphRef {
        GraphRef {
            data: Arc::downgrade(&self.data),
        }
    }
}

impl Clone for GraphRef {
    fn clone(&self) -> Self {
        GraphRef {
            data: self.data.clone(),
        }
    }
}
impl GraphRef {
    fn upgrade(self) -> Option<Graph> {
        if let Some(data) = self.data.upgrade() {
            Some(Graph { data })
        } else {
            None
        }
    }
}
