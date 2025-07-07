use crate::{operator::Operator, shape::tracker::ShapeTracker};
use std::fmt;

#[derive(Debug)]
pub struct Node {
    op: Box<dyn Operator>,
    pub shape: ShapeTracker,
}

impl Node {
    pub fn new(op: impl Operator + 'static, shape: ShapeTracker) -> Self {
        Self {
            op: Box::new(op),
            shape,
        }
    }

    pub fn op(&self) -> &dyn Operator {
        &*self.op
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}\n{:?}", self.op, self.shape)
    }
}
