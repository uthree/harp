use crate::prelude::*;

use std::{cell::RefCell, sync::Arc};
pub struct Tensor_ {
    graph: Graph,
    shape_tracker: ShapeTracker,
    operator: dyn Operator,
}

pub struct Tensor {
    content: Arc<RefCell<Tensor_>>,
}
