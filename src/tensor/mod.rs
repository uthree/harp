use std::cell::RefCell;
use std::rc::Rc;

use crate::backend::Buffer;

pub struct TensorData {
    buffer: Option<Box<dyn Buffer>>,
    requires_grad: bool,
}

pub struct Tensor(Rc<RefCell<TensorData>>);
