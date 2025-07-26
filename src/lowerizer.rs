//! # `Tensor` to `UOp` Lowering
//!
//! This module provides the `Lowerizer`, a component responsible for translating
//! a `Tensor`'s computation graph into a `UOp` graph. This process is often
//! referred to as "lowering," as it moves from a higher-level abstraction (`Tensor`)
//! to a lower-level, more explicit representation (`UOp`).

use crate::tensor::{Tensor, Tensor_, TensorOp};
use crate::uop::{Op, UOp};
use crate::dtype::DType;
use std::collections::HashMap;
use std::rc::Rc;

/// A struct that holds the state for the lowering process.
pub struct Lowerizer<'a, T> {
    // Maps a Tensor's memory address to its corresponding `UOp::Var` (e.g., "data0").
    arg_map: HashMap<*const Tensor_<T>, UOp>,
    phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T: 'a> Lowerizer<'a, T> {
    /// Creates a new `Lowerizer`.
    ///
    /// # Arguments
    ///
    /// * `kernel_args` - A slice of `Tensor`s that will be the arguments to the
    ///   generated kernel. The order of tensors in this slice determines the
    ///   index in the `bufs` array (e.g., `bufs[0]`, `bufs[1]`).
    pub fn new(kernel_args: &[&'a Tensor<T>]) -> Self {
        let mut arg_map = HashMap::new();
        for (i, tensor) in kernel_args.iter().enumerate() {
            let buffer = UOp::var(&format!("data{i}"), tensor.0.dtype.clone());
            arg_map.insert(Rc::as_ptr(&tensor.0), buffer);
        }
        Self {
            arg_map,
            phantom: std::marker::PhantomData,
        }
    }

    /// Lowers a `Tensor` computation graph to a `UOp` graph.
    ///
    /// This function traverses the `Tensor` graph, starting from the given root,
    /// and constructs an equivalent `UOp` graph that represents the computation.
    /// The final `UOp` will be a `Store` operation into the output buffer corresponding
    /// to the root `tensor`.
    pub fn lower(&mut self, tensor: &'a Tensor<T>) -> UOp {
        let loop_var = UOp::var("i", DType::U64);
        let result_expr = self.build_uop_graph(tensor, &loop_var);

        // The output buffer is the one that corresponds to the root tensor itself.
        let output_buffer = self.arg_map.get(&Rc::as_ptr(&tensor.0)).expect("Output buffer not found in arg_map").clone();
        let idx = tensor.0.tracker.expr_node(&loop_var);

        UOp::new(
            Op::Store,
            DType::Unit,
            vec![output_buffer, idx, result_expr],
        )
    }

    /// Recursively builds the `UOp` graph from the `Tensor` graph.
    fn build_uop_graph(&mut self, tensor: &'a Tensor<T>, loop_var: &UOp) -> UOp {
        match &tensor.0.op {
            TensorOp::Load => {
                // If it's a Load op, it must be one of the kernel arguments.
                let buffer = self.arg_map.get(&Rc::as_ptr(&tensor.0)).expect("Load buffer not found in arg_map").clone();
                let idx = tensor.0.tracker.expr_node(loop_var);
                UOp::new(Op::Load, tensor.0.dtype.clone(), vec![buffer, idx])
            }
            TensorOp::Unary(op) => {
                let src = self.build_uop_graph(&tensor.0.src[0], loop_var);
                UOp::new(op.clone(), tensor.0.dtype.clone(), vec![src])
            }
            TensorOp::Binary(op) => {
                let lhs = self.build_uop_graph(&tensor.0.src[0], loop_var);
                let rhs = self.build_uop_graph(&tensor.0.src[1], loop_var);
                UOp::new(op.clone(), tensor.0.dtype.clone(), vec![lhs, rhs])
            }
        }
    }
}
