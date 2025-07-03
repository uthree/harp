use crate::operator::Operator;
use crate::shape::ShapeTracker;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

#[derive(Debug)]
pub struct TensorNodeStore {
    shape_tracker: ShapeTracker,
    operator: Operator,
    inputs: Vec<TensorNode>,
}

pub type TensorNode = Weak<RefCell<TensorNodeStore>>;
