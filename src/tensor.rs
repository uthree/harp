use crate::graph::GraphRef;
use crate::ops::Operator;
use crate::prelude::ShapeTracker;

use std::{
    cell::RefCell,
    sync::{Arc, Weak},
};

// This struct holds the actual data and computation graph information for a tensor.
pub struct TensorData {
    pub graph: GraphRef,
    pub shape_tracker: ShapeTracker,
    pub inputs: Vec<Tensor>,
    pub operator: Box<dyn Operator>,
}

// A wrapper around TensorData that allows for multiple owners and weak references.
// It uses Arc<RefCell<>> to provide shared ownership and interior mutability.
#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<RefCell<TensorData>>,
}

// A weak reference to a Tensor, which doesn't prevent the Tensor from being dropped.
pub struct TensorRef {
    data: Weak<RefCell<TensorData>>,
}

impl Tensor {
    // Creates a weak reference (TensorRef) from a Tensor.
    pub fn downgrade(&self) -> TensorRef {
        TensorRef {
            data: Arc::downgrade(&self.data),
        }
    }

    pub fn graph(&self) -> GraphRef {
        self.data.borrow().graph.clone()
    }
}

impl TensorRef {
    // Attempts to upgrade a TensorRef to a Tensor.
    // Returns Some(Tensor) if the Tensor is still alive, otherwise None.
    pub fn upgrade(&self) -> Option<Tensor> {
        self.data.upgrade().map(|data| Tensor { data })
    }

    pub fn graph(&self) -> GraphRef {
        self.upgrade()
            .expect("TensorRef is dropped")
            .data
            .borrow()
            .graph
            .clone()
    }
}

impl std::ops::Add for TensorRef {
    type Output = TensorRef;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_tensor = self.upgrade().expect("Left-hand side tensor is dropped");
        let rhs_tensor = rhs.upgrade().expect("Right-hand side tensor is dropped");

        let graph = lhs_tensor
            .data
            .borrow()
            .graph
            .clone()
            .upgrade()
            .expect("Graph is dropped");

        let new_tensor_data = TensorData {
            graph: graph.clone().downgrade(),
            shape_tracker: lhs_tensor.data.borrow().shape_tracker.clone(), // Simplified: using lhs shape_tracker
            inputs: vec![lhs_tensor.clone(), rhs_tensor],
            operator: Box::new(crate::ops::Add {}),
        };

        let new_tensor = Tensor {
            data: std::sync::Arc::new(std::cell::RefCell::new(new_tensor_data)),
        };

        graph.data.borrow_mut().inter_nodes.push(new_tensor.clone());

        new_tensor.downgrade()
    }
}
