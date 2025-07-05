use crate::prelude::*;

use std::{cell::RefCell, sync::{Arc, Weak}};

pub struct TensorData {
    pub graph: Graph,
    pub shape_tracker: ShapeTracker,
    pub inputs: Vec<Tensor>,
    pub operator: Box<dyn Operator>,
}

'''#[derive(Clone)]
pub struct Tensor {
    content: Arc<RefCell<Tensor_>>,
}

pub struct TensorRef {
    content: Weak<RefCell<Tensor_>>,
}

impl Tensor {
    pub fn downgrade(&self) -> TensorRef {
        TensorRef {
            content: Arc::downgrade(&self.content),
        }
    }
}

impl TensorRef {
    pub fn upgrade(&self) -> Option<Tensor> {
        self.content.upgrade().map(|content| Tensor { content })
    }
}''

pub struct TensorRef {}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
        }
    }
}
