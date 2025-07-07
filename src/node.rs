use crate::operator::Operator;

#[derive(Debug)]
pub struct Node {
    op: Box<dyn Operator>,
}

impl Node {
    pub fn new(op: impl Operator + 'static) -> Self {
        Self { op: Box::new(op) }
    }

    pub fn op(&self) -> &dyn Operator {
        &*self.op
    }
}
