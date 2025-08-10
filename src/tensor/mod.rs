use std::cell::RefCell;
use std::rc::Rc;

use crate::cbuffer::CBuffer;

pub enum TensorBuffer {
    C(CBuffer),
}

pub struct TensorData {
    grad: Option<Tensor>,
    requires_grad: bool,
}

pub struct Tensor(Rc<RefCell<TensorData>>);
