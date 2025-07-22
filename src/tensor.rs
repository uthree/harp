use crate::backends::{Backend, Variable};
use crate::dtype::DType;
use crate::uop::UOp;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

pub struct Tensor_ {
    pub op: UOp,
    pub src: Vec<Tensor>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub backend: Arc<dyn Backend>,
    pub realized: RefCell<Option<Variable>>,
}

#[derive(Clone)]
pub struct Tensor(pub Rc<Tensor_>);

impl Tensor {
    pub fn new(
        op: UOp,
        src: Vec<Tensor>,
        shape: Vec<usize>,
        dtype: DType,
        backend: Arc<dyn Backend>,
    ) -> Self {
        Self(Rc::new(Tensor_ {
            op,
            src,
            shape,
            dtype,
            backend,
            realized: RefCell::new(None),
        }))
    }

    pub fn realize(&self) -> Variable {
        if let Some(ref realized) = *self.0.realized.borrow() {
            return realized.clone();
        }

        // TODO: ここで実際にUOpグラフを辿ってコンパイル・実行する
        // self.0.backend.compile_and_exec(...);

        // 仮のVariableを返す
        let dummy_var = self.0.backend.alloc(1, self.0.backend.clone());
        *self.0.realized.borrow_mut() = Some(dummy_var.clone());
        dummy_var
    }
}