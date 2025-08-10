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

impl Clone for TensorBackend {
    fn clone(&self) -> Self {
        match self {
            TensorBackend::C(backend) => TensorBackend::C(backend.clone()),
        }
    }
}

pub enum TensorOp {
    Rand(Shape),
    Add,
}

pub struct TensorData {
    op: TensorOp,
    src: Vec<Tensor>,
    grad: Option<Tensor>,
    requires_grad: bool,
    backend: TensorBackend,
}

pub type Shape = Vec<usize>;

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    pub fn rand(shape: Shape) -> Self {
        TensorData {
            op: TensorOp::Rand(shape),
            src: vec![],
            grad: None,
            requires_grad: false,
            backend: backend("c"),
        }
        .into()
    }
}

impl PartialEq for TensorBackend {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorBackend::C(a), TensorBackend::C(b)) => Rc::ptr_eq(a, b),
        }
    }
}

impl std::ops::Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let self_backend = &self.0.borrow().backend;
        let rhs_backend = &rhs.0.borrow().backend;
        if self_backend != rhs_backend {
            panic!("Backends of tensors do not match");
        }
        TensorData {
            op: TensorOp::Add,
            src: vec![self.clone(), rhs.clone()],
            grad: None,
            requires_grad: false,
            backend: self.0.borrow().backend.clone(),
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
