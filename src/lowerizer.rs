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
pub struct Lowerizer<T> {
    arg_map: HashMap<*const Tensor_<T>, UOp>,
}

impl<T> Default for Lowerizer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Lowerizer<T> {
    /// Creates a new `Lowerizer`.
    pub fn new() -> Self {
        Self {
            arg_map: HashMap::new(),
        }
    }

    /// Lowers a `Tensor` computation graph to a `UOp` graph.
    ///
    /// This function traverses the `Tensor` graph, starting from the given root,
    /// and constructs an equivalent `UOp` graph that represents the computation.
    pub fn lower(&mut self, tensor: &Tensor<T>) -> UOp {
        let loop_var = UOp::var("i", DType::U64);
        let result_expr = self.build_uop_graph(tensor, &loop_var);

        let out_idx = self.arg_map.len();
        let output_buffer = UOp::var(&format!("data{out_idx}"), tensor.0.dtype.clone());
        let idx = tensor.0.tracker.expr_node(&loop_var);

        UOp::new(
            Op::Store,
            DType::Unit,
            vec![output_buffer, idx, result_expr],
        )
    }

    /// Recursively builds the `UOp` graph from the `Tensor` graph.
    fn build_uop_graph(&mut self, tensor: &Tensor<T>, loop_var: &UOp) -> UOp {
        let tensor_ptr = Rc::as_ptr(&tensor.0);
        if let Some(uop) = self.arg_map.get(&tensor_ptr) {
            return uop.clone();
        }

        let uop = match &tensor.0.op {
            TensorOp::Load => {
                let buffer_name = format!("data{}", self.arg_map.len());
                let buffer = UOp::var(&buffer_name, tensor.0.dtype.clone());
                let idx = tensor.0.tracker.expr_node(loop_var);
                UOp::new(Op::Load, tensor.0.dtype.clone(), vec![buffer, idx])
            }
            TensorOp::Binary(op) => {
                let lhs = self.build_uop_graph(&tensor.0.src[0], loop_var);
                let rhs = self.build_uop_graph(&tensor.0.src[1], loop_var);
                UOp::new(op.clone(), tensor.0.dtype.clone(), vec![lhs, rhs])
            }
        };

        self.arg_map.insert(tensor_ptr, uop.clone());
        uop
    }
}