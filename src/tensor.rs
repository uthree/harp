use crate::graph::Graph;
use crate::ops::Operator;
use crate::prelude::ShapeTracker;

use std::{
    cell::RefCell,
    sync::{Arc, Weak},
};

// This struct holds the actual data and computation graph information for a tensor.
pub struct TensorData {
    pub graph: Graph,
    pub shape_tracker: ShapeTracker,
    pub inputs: Vec<Tensor>,
    pub operator: Box<dyn Operator>,
}

// A wrapper around TensorData that allows for multiple owners and weak references.
// It uses Arc<RefCell<>> to provide shared ownership and interior mutability.
#[derive(Clone)]
pub struct Tensor {
    pub content: Arc<RefCell<TensorData>>,
}

// A weak reference to a Tensor, which doesn't prevent the Tensor from being dropped.
pub struct TensorRef {
    content: Weak<RefCell<TensorData>>,
}

impl Tensor {
    // Creates a weak reference (TensorRef) from a Tensor.
    pub fn downgrade(&self) -> TensorRef {
        TensorRef {
            content: Arc::downgrade(&self.content),
        }
    }
}

impl TensorRef {
    // Attempts to upgrade a TensorRef to a Tensor.
    // Returns Some(Tensor) if the Tensor is still alive, otherwise None.
    pub fn upgrade(&self) -> Option<Tensor> {
        self.content.upgrade().map(|content| Tensor { content })
    }
}