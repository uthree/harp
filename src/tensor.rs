use std::{ops::Deref, rc::Rc};

#[derive(Debug, Clone)]
struct TensorData {}

#[derive(Debug, Clone)]
struct Tensor(Rc<TensorData>);

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for Tensor {
    type Target = TensorData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
