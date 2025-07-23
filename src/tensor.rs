use crate::backends::{Backend, Variable};
use crate::dot::ToDot;
use crate::dtype::{DType, IsNumber};
use crate::lower;
use crate::optimizer::Optimizer;
use crate::shapetracker::ShapeTracker;
use crate::uop::{Op, UOp};
use log::debug;
use ndarray::{Array, ArrayD};
use rustc_hash::{FxHashMap, FxHashSet};
use std::any::Any;
use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum TensorOp {
    Load,
    Binary(Op),
}

pub struct Tensor_ {
    op: TensorOp,
    src: Vec<Tensor>,
    pub tracker: ShapeTracker,
    pub dtype: DType,
    pub backend: Arc<dyn Backend>,
    variable: Option<Variable>,
}

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<Tensor_>>);

impl Tensor {
    pub fn new(
        op: TensorOp,
        src: Vec<Tensor>,
        tracker: ShapeTracker,
        dtype: DType,
        backend: Arc<dyn Backend>,
        variable: Option<Variable>,
    ) -> Self {
        Self(Rc::new(RefCell::new(Tensor_ {
            op,
            src,
            tracker,
            dtype,
            backend,
            variable,
        })))
    }

    pub fn realize(&self) -> Variable {
        if let Some(ref realized) = self.0.borrow().variable {
            return realized.clone();
        }

        let (loop_op, mut kernel_args) = {
            let self_borrow = self.0.borrow();
            self.to_uop(&self_borrow)
        };
        debug!("Generated UOp graph: {:?}", loop_op);
        debug!("Collected kernel args: {:?}", kernel_args);

        let output_buffer = {
            let self_borrow = self.0.borrow();
            let n_elements = self_borrow.tracker.shape().iter().product::<usize>();
            self_borrow.backend.alloc(
                n_elements * self_borrow.dtype.size(),
                self_borrow.dtype,
            )
        };
        kernel_args.push(output_buffer.clone());

        let optimizer = Optimizer::new();
        let optimized_loop_op = optimizer.optimize(&loop_op);
        debug!("Optimized UOp graph: {:?}", optimized_loop_op);

        let ast = lower::lower(&optimized_loop_op);
        let args_ref: Vec<&Variable> = kernel_args.iter().collect();

        self.0
            .borrow()
            .backend
            .compile_and_exec(&ast, &args_ref);

        self.0.borrow_mut().variable = Some(output_buffer.clone());
        output_buffer
    }

    fn to_uop(&self, self_borrow: &Tensor_) -> (UOp, Vec<Variable>) {
        let mut arg_vars: Vec<Variable> = Vec::new();
        let mut arg_map: FxHashMap<*const RefCell<Tensor_>, UOp> = FxHashMap::default();

        fn build_uop_graph(
            tensor: &Tensor,
            arg_vars: &mut Vec<Variable>,
            arg_map: &mut FxHashMap<*const RefCell<Tensor_>, UOp>,
            loop_var: &UOp,
        ) -> UOp {
            let ptr = Rc::as_ptr(&tensor.0);
            if let Some(uop) = arg_map.get(&ptr) {
                return uop.clone();
            }

            let t_borrow = tensor.0.borrow();
            let uop = match &t_borrow.op {
                TensorOp::Load => {
                    let var = t_borrow
                        .variable
                        .as_ref()
                        .expect("Leaf tensor must have a variable")
                        .clone();
                    let buffer =
                        UOp::var(&format!("data{}", arg_vars.len()), t_borrow.dtype.clone());
                    arg_vars.push(var);
                    let idx = t_borrow.tracker.expr_indices(loop_var.clone());
                    UOp::new(Op::Load, t_borrow.dtype.clone(), vec![buffer, idx])
                }
                TensorOp::Binary(op) => {
                    let lhs = build_uop_graph(&t_borrow.src[0], arg_vars, arg_map, loop_var);
                    let rhs = build_uop_graph(&t_borrow.src[1], arg_vars, arg_map, loop_var);
                    UOp::new(op.clone(), t_borrow.dtype.clone(), vec![lhs, rhs])
                }
            };
            arg_map.insert(ptr, uop.clone());
            uop
        }

        let loop_var = UOp::var("i", DType::I32);
        let result_expr = build_uop_graph(self, &mut arg_vars, &mut arg_map, &loop_var);

        let out_idx = arg_vars.len();
        let output_buffer = UOp::var(&format!("data{}", out_idx), self_borrow.dtype.clone());
        let idx = self_borrow.tracker.expr_indices(loop_var);
        let store = UOp::new(
            Op::Store,
            DType::Unit,
            vec![output_buffer, idx, result_expr],
        );

        let n_elements = self_borrow.tracker.shape().iter().product::<usize>();
        let loop_op = UOp::new(Op::Loop, DType::Unit, vec![(n_elements as u64).into(), store]);

        (loop_op, arg_vars)
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let self_borrow = self.0.borrow();
        let new_tracker = self_borrow.tracker.reshape(new_shape);
        Self::new(
            self_borrow.op.clone(),
            self_borrow.src.clone(),
            new_tracker,
            self_borrow.dtype.clone(),
            self_borrow.backend.clone(),
            None,
        )
    }

    fn lazy_binary_op(op: Op, a: &Self, b: &Self) -> Self {
        let a_borrow = a.0.borrow();
        let b_borrow = b.0.borrow();
        assert!(Arc::ptr_eq(&a_borrow.backend, &b_borrow.backend));
        Self::new(
            TensorOp::Binary(op),
            vec![a.clone(), b.clone()],
            a_borrow.tracker.clone(),
            a_borrow.dtype.clone(),
            a_borrow.backend.clone(),
            None,
        )
    }

    pub fn from_vec<T: IsNumber + 'static>(
        data: Vec<T>,
        shape: Vec<usize>,
        backend: Arc<dyn Backend>,
    ) -> Self {
        let tracker = ShapeTracker::new(shape);
        assert_eq!(tracker.shape().iter().product::<usize>(), data.len());
        let variable = backend.copy_to_device(&data, T::dtype());
        Self::new(
            TensorOp::Load,
            vec![],
            tracker,
            T::dtype(),
            backend,
            Some(variable),
        )
    }

    pub fn from_ndarray<T: IsNumber + 'static>(
        arr: &ArrayD<T>,
        backend: Arc<dyn Backend>,
    ) -> Self {
        let shape = arr.shape().to_vec();
        let data = arr.iter().cloned().collect::<Vec<T>>();
        Self::from_vec(data, shape, backend)
    }

    pub fn to_vec<T: IsNumber + 'static>(&self) -> Vec<T> {
        let realized_var = self.realize();
        let self_borrow = self.0.borrow();
        assert_eq!(self_borrow.dtype, T::dtype());
        let any_vec = self_borrow.backend.copy_from_device(&realized_var);
        *any_vec.downcast::<Vec<T>>().unwrap()
    }

    pub fn to_ndarray<T: IsNumber + 'static>(&self) -> ArrayD<T> {
        let shape = self.0.borrow().tracker.shape().to_vec();
        let data = self.to_vec::<T>();
        Array::from_shape_vec(shape, data).unwrap()
    }
}

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

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::lazy_binary_op(Op::Mul, self, rhs)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl ToDot for Tensor {
    fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph G {\n");
        dot.push_str("  node [shape=box];\n");
        let mut visited = FxHashSet::default();
        build_dot_tensor(self, &mut dot, &mut visited);
        dot.push_str("}\n");
        dot
    }
}

fn build_dot_tensor(
    tensor: &Tensor,
    dot: &mut String,
    visited: &mut FxHashSet<*const RefCell<Tensor_>>,
) {
    let ptr = Rc::as_ptr(&tensor.0);
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    let t_borrow = tensor.0.borrow();
    let label = format!(
        "op: {:?}\\nshape: {:?}\\ndtype: {:?}",
        t_borrow.op,
        t_borrow.tracker.shape(),
        t_borrow.dtype
    );
    dot.push_str(&format!("  \"{ptr:p}\" [label=\"{label}\"];\n"));

    for src in &t_borrow.src {
        let src_ptr = Rc::as_ptr(&src.0);
        dot.push_str(&format!("  \"{src_ptr:p}\" -> \"{ptr:p}\";\n"));
        build_dot_tensor(src, dot, visited);
    }
}