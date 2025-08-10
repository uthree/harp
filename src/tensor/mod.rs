use crate::backend::Backend;
use crate::backend::c::CBackend;
use crate::cbuffer::CBuffer;
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    // Store an Rc in the thread local, so we can share it.
    static C_BACKEND: Rc<RefCell<CBackend>> = Rc::new(RefCell::new(CBackend::new()));
}

pub enum TensorBuffer {
    C(CBuffer),
}
pub enum TensorBackend {
    C(Rc<RefCell<CBackend>>),
}

pub enum TensorOp {
    Rand(Shape),
    Add(Tensor, Tensor),
}

pub struct TensorData {
    op: TensorOp,
    grad: Option<Tensor>,
    requires_grad: bool,
    backend: TensorBackend,
}

pub type Shape = Vec<usize>;

pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    pub fn rand(shape: Shape) -> Self {
        TensorData {
            op: TensorOp::Rand(shape),
            grad: None,
            requires_grad: false,
            backend: backend("c"),
        }
        .into()
    }
}

impl From<TensorData> for Tensor {
    fn from(value: TensorData) -> Self {
        Tensor(Rc::new(RefCell::new(value)))
    }
}

// By using C_BACKEND.with, we get a reference to the thread-local value
// and return a cloned Rc.
pub fn backend(name: &str) -> TensorBackend {
    match name {
        "c" => C_BACKEND.with(|backend| TensorBackend::C(backend.clone())),
        _ => panic!("Unsupported backend: {}", name),
    }
}
