use std::cell::RefCell;

use crate::backend::Backend;
use crate::backend::c::CBackend;
use crate::cbuffer::CBuffer;

thread_local! {
    pub static C_BACKEND: RefCell<CBackend> = RefCell::new(CBackend::new());
}

pub enum TensorBuffer {
    C(CBuffer),
}

pub enum TensorBackend {
    C(CBackend),
}

pub struct TensorData {
    grad: Option<Tensor>,
    requires_grad: bool,
}

pub struct Tensor(std::rc::Rc<RefCell<TensorData>>);
