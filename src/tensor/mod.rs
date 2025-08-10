use crate::ast::DType;
use crate::backend::c::CBackend;
use crate::backend::Backend;
use crate::cbuffer::CBuffer;
use crate::graph::Graph;
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    static C_BACKEND: Rc<RefCell<CBackend>> = Rc::new(RefCell::new(CBackend::new()));
}

pub enum TensorBuffer {
    C(CBuffer),
}

#[derive(Clone)]
pub enum TensorBackend {
    C(Rc<RefCell<CBackend>>),
}

impl PartialEq for TensorBackend {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorBackend::C(a), TensorBackend::C(b)) => Rc::ptr_eq(a, b),
        }
    }
}

#[derive(Clone)]
pub enum TensorOp {
    Rand,
    Add,
}

pub struct TensorData {
    op: TensorOp,
    src: Vec<Tensor>,
    shape: Shape,
    dtype: DType,
    buffer: Option<TensorBuffer>,
    grad: Option<Tensor>,
    requires_grad: bool,
    backend: TensorBackend,
}

pub type Shape = Vec<usize>;

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    pub fn rand(shape: Shape, dtype: DType) -> Self {
        TensorData {
            op: TensorOp::Rand,
            src: vec![],
            shape,
            dtype,
            buffer: None,
            grad: None,
            requires_grad: false,
            backend: backend("c"),
        }
        .into()
    }

    pub fn forward(&self) {
        if self.0.borrow().buffer.is_some() {
            return;
        }

        for s in &self.0.borrow().src {
            s.forward();
        }

        let graph = Graph::new();
        let data = self.0.borrow();

        let srcs: Vec<_> = data
            .src
            .iter()
            .map(|s| {
                let s_data = s.0.borrow();
                graph.input(s_data.dtype.clone(), s_data.shape.iter().map(|d| (*d).into()).collect())
            })
            .collect();

        let op = match data.op.clone() {
            TensorOp::Rand => {
                graph.rand(data.dtype.clone(), data.shape.iter().map(|d| (*d).into()).collect())
            }
            TensorOp::Add => srcs[0].clone() + srcs[1].clone(),
        };

        op.as_output();

        let result_buffer = match data.backend.clone() {
            TensorBackend::C(b) => TensorBuffer::C(b.borrow().run(&graph)),
        };

        drop(data);
        self.0.borrow_mut().buffer = Some(result_buffer);
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
        if self.0.borrow().dtype != rhs.0.borrow().dtype {
            panic!("Dtypes of tensors do not match");
        }
        let shape = self.0.borrow().shape.clone();
        let dtype = self.0.borrow().dtype.clone();
        TensorData {
            op: TensorOp::Add,
            src: vec![self.clone(), rhs.clone()],
            shape,
            dtype,
            buffer: None,
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

pub fn backend(name: &str) -> TensorBackend {
    match name {
        "c" => C_BACKEND.with(|backend| TensorBackend::C(backend.clone())),
        _ => panic!("Unsupported backend: {}", name),
    }
}
