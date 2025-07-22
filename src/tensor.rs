use crate::backends::{Backend, Variable};
use crate::dtype::DType;
use crate::lower;
use crate::uop::{Op, UOp};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

pub enum TensorOp {
    Load,
    Binary(Op),
}

pub struct Tensor_ {
    pub op: TensorOp,
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
        op: TensorOp,
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

        let result_var = match self.0.op {
            TensorOp::Load => {
                // For a Load op, we just need to allocate the buffer
                let size: usize = self.0.shape.iter().product::<usize>() * self.0.dtype.size();
                self.0.backend.alloc(size, self.0.backend.clone())
            }
            TensorOp::Binary(_) => {
                // Realize the source tensors
                let mut args: Vec<_> = self.0.src.iter().map(|t| t.realize()).collect();

                // Allocate the output buffer for this operation
                let size: usize = self.0.shape.iter().product::<usize>() * self.0.dtype.size();
                let output_buffer = self.0.backend.alloc(size, self.0.backend.clone());
                
                // The output buffer is also an argument to the kernel
                let mut kernel_args = args;
                kernel_args.push(output_buffer.clone());

                // Convert the Tensor graph to a UOp graph
                let uop = self.to_uop();
                let ast = lower::lower(&uop);

                // Compile and execute
                let args_ref: Vec<&Variable> = kernel_args.iter().collect();
                self.0.backend.compile_and_exec(&ast, &args_ref);

                output_buffer
            }
        };

        *self.0.realized.borrow_mut() = Some(result_var.clone());
        result_var
    }

    fn to_uop(&self) -> UOp {
        // A simple recursive function to build the UOp graph from the Tensor graph
        fn build_uop_graph(tensor: &Tensor, arg_map: &mut std::collections::HashMap<*const Tensor_, usize>) -> UOp {
            match &tensor.0.op {
                TensorOp::Load => {
                    let ptr = Rc::as_ptr(&tensor.0);
                    let next_index = arg_map.len();
                    let arg_index = *arg_map.entry(ptr).or_insert(next_index);
                    UOp::var(&format!("data{}", arg_index), tensor.0.dtype.clone())
                }
                TensorOp::Binary(op) => {
                    let lhs = build_uop_graph(&tensor.0.src[0], arg_map);
                    let rhs = build_uop_graph(&tensor.0.src[1], arg_map);
                    UOp::new(op.clone(), tensor.0.dtype.clone(), vec![lhs, rhs])
                }
            }
        }

        let mut arg_map = std::collections::HashMap::new();
        let result_expr = build_uop_graph(self, &mut arg_map);

        // The final operation in a kernel is always a store.
        // The output buffer is the last argument.
        let out_idx = arg_map.len();
        let output_buffer = UOp::var(&format!("data{}", out_idx), self.0.dtype.clone());
        let loop_var = UOp::var("i", DType::U64);
        let store = UOp::new(Op::Store, DType::Unit, vec![output_buffer, loop_var.clone(), result_expr]);

        // Create a loop around the store
        let n_elements = self.0.shape.iter().product::<usize>();
        UOp::new(Op::Loop, DType::Unit, vec![(n_elements as u64).into(), store])
    }

    fn lazy_binary_op(op: Op, a: &Self, b: &Self) -> Self {
        // TODO: shapeやdtypeのチェック・プロモーション
        assert!(Arc::ptr_eq(&a.0.backend, &b.0.backend));
        Self::new(
            TensorOp::Binary(op),
            vec![a.clone(), b.clone()],
            a.0.shape.clone(),
            a.0.dtype.clone(),
            a.0.backend.clone(),
        )
    }
}

use std::ops::Add;

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Add, self, rhs)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}