//! # `Tensor` to `UOp` Lowering
//!
//! This module provides the `Lowerizer`, a component responsible for translating
//! a `Tensor`'s computation graph into a `UOp` graph. This process is often
//! referred to as "lowering," as it moves from a higher-level abstraction (`Tensor`)
//! to a lower-level, more explicit representation (`UOp`).

use crate::dtype::DType;
use crate::tensor::{Tensor, Tensor_, TensorOp};
use crate::uop::{Op, UOp};
use std::collections::HashMap;
use std::rc::Rc;

/// A struct that holds the state for the lowering process.
pub struct Lowerizer<'a> {
    // Maps a Tensor's memory address to its corresponding `UOp::Var` (e.g., "data0").
    arg_map: HashMap<*const Tensor_, UOp>,
    phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Lowerizer<'a> {
    /// Creates a new `Lowerizer`.
    ///
    /// # Arguments
    ///
    /// * `kernel_args` - A slice of `Tensor`s that will be the arguments to the
    ///   generated kernel. The order of tensors in this slice determines the
    ///   index in the `bufs` array (e.g., `bufs[0]`, `bufs[1]`).
    pub fn new(kernel_arg_tensors: &'a [&'a Tensor]) -> Self {
        let mut arg_map = HashMap::new();
        for (i, t) in kernel_arg_tensors.iter().enumerate() {
            let ptr = Rc::as_ptr(&t.0);
            let uop_var = UOp::var(&format!("data{i}"), t.dtype.clone());
            arg_map.insert(ptr, uop_var);
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
    pub fn lower(&mut self, tensor: &'a Tensor) -> UOp {
        match &tensor.0.op {
            TensorOp::Reduce(axis, op) => {
                let src_tensor = &tensor.0.src[0];
                let acc_var = UOp::var("acc", tensor.0.dtype.clone());

                // 1. Declare and initialize accumulator
                let identity = op
                    .identity_element(&tensor.0.dtype)
                    .expect("Identity element not found for reduce op");
                let declare_acc = UOp::new(
                    Op::Declare("acc".to_string(), tensor.0.dtype.clone()),
                    DType::Unit,
                    vec![identity.into()],
                );

                // 2. Create the reduction loop
                let reduce_loop_var = UOp::var("ridx", DType::U64);
                let reduce_dim = src_tensor.shape()[*axis];
                let loop_start = UOp::new(
                    Op::LoopStart,
                    DType::Unit,
                    vec![reduce_loop_var.clone(), (reduce_dim as u64).into()],
                );

                // 3. Build the expression to load the source element
                let mut src_indices: Vec<UOp> = tensor
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(i, _)| UOp::var(&format!("idx{i}"), DType::U64))
                    .collect();
                src_indices.insert(*axis, reduce_loop_var);

                let src_idx_expr = src_tensor.0.tracker.expr_indices(Some(&src_indices));
                let src_buffer = self
                    .arg_map
                    .get(&Rc::as_ptr(&src_tensor.0))
                    .unwrap()
                    .clone();
                let load_expr = UOp::new(
                    Op::Load,
                    src_tensor.0.dtype.clone(),
                    vec![src_buffer, src_idx_expr],
                );

                // 4. Create the accumulation and reassignment
                let acc_op = UOp::new(
                    op.clone(),
                    tensor.0.dtype.clone(),
                    vec![acc_var.clone(), load_expr],
                );
                let update_acc = UOp::new(Op::Store, DType::Unit, vec![acc_var.clone(), acc_op]);

                // 5. End the loop
                let loop_end = UOp::new(Op::LoopEnd, DType::Unit, vec![]);

                // 6. Store the final result
                let output_buffer = self.arg_map.get(&Rc::as_ptr(&tensor.0)).unwrap().clone();
                let output_indices: Vec<UOp> = tensor
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(i, _)| UOp::var(&format!("idx{i}"), DType::U64))
                    .collect();
                let output_idx = tensor.0.tracker.expr_indices(Some(&output_indices));
                let store_result = UOp::new(
                    Op::Store,
                    DType::Unit,
                    vec![output_buffer, output_idx, acc_var],
                );

                UOp::new(
                    Op::Block,
                    DType::Unit,
                    vec![declare_acc, loop_start, update_acc, loop_end, store_result],
                )
            }
            TensorOp::Scan { axis, op } => {
                let src_tensor = &tensor.0.src[0];
                let acc_var = UOp::var("acc", tensor.0.dtype.clone());

                // 1. Declare and initialize accumulator
                let identity = op
                    .identity_element(&tensor.0.dtype)
                    .expect("Identity element not found for scan op");
                let declare_acc = UOp::new(
                    Op::Declare("acc".to_string(), tensor.0.dtype.clone()),
                    DType::Unit,
                    vec![identity.into()],
                );

                // 2. Create the scan loop
                let scan_loop_var = UOp::var("sidx", DType::U64);
                let scan_dim = src_tensor.shape()[*axis];
                let loop_start = UOp::new(
                    Op::LoopStart,
                    DType::Unit,
                    vec![scan_loop_var.clone(), (scan_dim as u64).into()],
                );

                // 3. Build expressions for loading source and storing to destination
                let loop_indices: Vec<UOp> = tensor
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        if i == *axis {
                            scan_loop_var.clone()
                        } else {
                            UOp::var(&format!("idx{i}"), DType::U64)
                        }
                    })
                    .collect();

                let src_idx_expr = src_tensor.0.tracker.expr_indices(Some(&loop_indices));
                let src_buffer = self
                    .arg_map
                    .get(&Rc::as_ptr(&src_tensor.0))
                    .unwrap()
                    .clone();
                let load_expr = UOp::new(
                    Op::Load,
                    src_tensor.0.dtype.clone(),
                    vec![src_buffer, src_idx_expr],
                );

                // 4. Create the accumulation and reassignment
                let acc_op = UOp::new(
                    op.clone(),
                    tensor.0.dtype.clone(),
                    vec![acc_var.clone(), load_expr],
                );
                let update_acc = UOp::new(Op::Store, DType::Unit, vec![acc_var.clone(), acc_op]);

                // 5. Store the intermediate result in the output buffer
                let output_buffer = self.arg_map.get(&Rc::as_ptr(&tensor.0)).unwrap().clone();
                let output_idx = tensor.0.tracker.expr_indices(Some(&loop_indices));
                let store_intermediate = UOp::new(
                    Op::Store,
                    DType::Unit,
                    vec![output_buffer, output_idx, acc_var.clone()],
                );

                // 6. End the loop
                let loop_end = UOp::new(Op::LoopEnd, DType::Unit, vec![]);

                UOp::new(
                    Op::Block,
                    DType::Unit,
                    vec![
                        declare_acc,
                        loop_start,
                        update_acc,
                        store_intermediate,
                        loop_end,
                    ],
                )
            }
            _ => {
                let loop_var = UOp::var("i", DType::U64);
                let result_expr = self.build_uop_graph(tensor, &loop_var);

                let output_buffer = self
                    .arg_map
                    .get(&Rc::as_ptr(&tensor.0))
                    .expect("Output buffer not found in arg_map")
                    .clone();
                let idx = tensor.0.tracker.expr_node(&loop_var);

                UOp::new(
                    Op::Store,
                    DType::Unit,
                    vec![output_buffer, idx, result_expr],
                )
            }
        }
    }

    /// Recursively builds the `UOp` graph from the `Tensor` graph.
    fn build_uop_graph(&mut self, tensor: &'a Tensor, loop_var: &UOp) -> UOp {
        match &tensor.0.op {
            TensorOp::Load => {
                let buffer = self
                    .arg_map
                    .get(&Rc::as_ptr(&tensor.0))
                    .expect("Load buffer not found in arg_map")
                    .clone();
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
            TensorOp::Reduce(_, _) => {
                panic!("Reduce should be handled in lower, not build_uop_graph");
            }
            TensorOp::Scan { .. } => {
                panic!("Scan should be handled in lower, not build_uop_graph");
            }
            TensorOp::Constant(n) => UOp::new(Op::Const(n.clone()), tensor.0.dtype.clone(), vec![]),
        }
    }
}
