use crate::prelude::*;

use std::{cell::RefCell, sync::Arc};

pub struct TensorData {
    pub graph: Graph,
    pub shape_tracker: ShapeTracker,
    pub inputs: Vec<Tensor>,
    pub operator: Box<dyn Operator>,
}

pub struct Tensor {
    pub data: Arc<RefCell<TensorData>>,
}

pub struct TensorRef {}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
        }
    }
}
