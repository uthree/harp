use crate::prelude::*;

use std::{cell::RefCell, sync::Arc};

pub struct TensorData {
    graph: Graph,
    shape_tracker: ShapeTracker,
    inputs: Vec<Tensor>,
    operator: Box<dyn Operator>,
}

pub struct Tensor {
    data: Arc<RefCell<TensorData>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
        }
    }
}
