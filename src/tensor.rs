use std::{ops::Deref, rc::Rc};

#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Rc<TensorData>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}
