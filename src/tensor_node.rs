use crate::operator::{Operator, Primitive};
use crate::shape::ShapeTracker;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

#[derive(Debug)]
pub struct TensorNodeStore {
    shape_tracker: ShapeTracker,
    operator: Box<dyn Operator>,
}

pub type TensorNode = Weak<RefCell<TensorNodeStore>>;
