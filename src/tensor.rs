use crate::prelude::*;

use std::{cell::RefCell, sync::Arc};

pub struct Tensor_ {
    graph: Graph,
    shape_tracker: ShapeTracker,
    inputs: Vec<Tensor>,
    operator: Box<dyn Operator>,
}

pub struct Tensor {
    content: Arc<RefCell<Tensor_>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            content: self.content.clone(),
        }
    }
}
