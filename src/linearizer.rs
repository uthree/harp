//! # `UOp` Linearizer
//!
//! This module implements the linearization process, which transforms a `UOp`
//! computation graph into a flat, sequential list of `UOp` instructions. This
//! linear representation is ideal for code generation, as the renderer can simply
//! iterate through the list and emit the corresponding code for each instruction.
//!
//! The linearizer handles control flow, such as loops, by introducing special
//! `Op::LoopStart` and `Op::LoopEnd` instructions into the instruction stream.

use crate::uop::{Op, UOp};
use log::debug;
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// A structure that holds the state for the linearization process.
pub struct Linearizer {
    node_map: FxHashMap<*const crate::uop::UOp_, UOp>,
    kernel_body: Vec<UOp>,
    var_counter: usize,
    loop_var_counter: usize,
}

impl Default for Linearizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Linearizer {
    /// Creates a new `Linearizer`.
    pub fn new() -> Self {
        Self {
            node_map: FxHashMap::default(),
            kernel_body: Vec::new(),
            var_counter: 0,
            loop_var_counter: 0,
        }
    }

    /// Generates a new unique variable name with a given prefix.
    fn new_var(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.var_counter);
        self.var_counter += 1;
        name
    }

    /// Generates a new unique loop variable name (e.g., "lidx0", "lidx1").
    fn new_loop_var(&mut self) -> UOp {
        let name = format!("lidx{}", self.loop_var_counter);
        self.loop_var_counter += 1;
        UOp::var(&name, crate::dtype::DType::U64)
    }

    /// Processes a `UOp` node from the computation graph and adds its linearized
    /// form to the kernel body.
    ///
    /// This method recursively traverses the graph, converting expression subtrees
    /// into temporary variables and storing the assignments in the `kernel_body`.
    /// It uses a map (`node_map`) to cache results and avoid redundant processing
    /// of shared nodes.
    fn process_node(&mut self, node: &UOp) -> UOp {
        let node_ptr = Rc::as_ptr(&node.0);
        if let Some(mapped) = self.node_map.get(&node_ptr) {
            return mapped.clone();
        }

        let new_srcs: Vec<UOp> = node
            .0
            .src
            .iter()
            .map(|src| self.process_node(src))
            .collect();

        let result_uop = match &node.0.op {
            Op::Const(_) | Op::Var(_) => {
                return UOp::new(node.0.op.clone(), node.0.dtype.clone(), new_srcs);
            }
            _ => {
                let var_name = self.new_var("var");
                let var_dtype = node.0.dtype.clone();
                let var_uop = UOp::var(&var_name, var_dtype.clone());
                let expr = UOp::new(node.0.op.clone(), var_dtype, new_srcs);

                let store_stmt = UOp::new(
                    Op::Store,
                    crate::dtype::DType::Unit,
                    vec![var_uop.clone(), expr],
                );
                self.kernel_body.push(store_stmt);
                var_uop
            }
        };

        self.node_map.insert(node_ptr, result_uop.clone());
        result_uop
    }

    /// Linearizes the entire computation graph starting from a root `UOp`.
    ///
    /// This function takes the final `UOp` of a computation graph (which is
    /// typically a `Store` operation) and the tensor shape, then transforms the
    /// entire graph into a single, flat `Vec<UOp>` representing the kernel.
    pub fn linearize(&mut self, root: &UOp, shape: &[usize]) -> Vec<UOp> {
        debug!("Linearizing UOp graph: {root:?}");

        // 1. Create loop variable and limit.
        let loop_var = self.new_loop_var();
        let n_elements: usize = shape.iter().product();
        let loop_limit = UOp::from(n_elements as u64);

        // 2. Start the kernel with a `LoopStart` instruction.
        let mut linearized_kernel = vec![UOp::new(
            Op::LoopStart,
            crate::dtype::DType::Unit,
            vec![loop_var.clone(), loop_limit],
        )];

        // 3. The root of the graph must be a `Store` operation.
        assert!(
            matches!(root.0.op, Op::Store),
            "The root of a computation graph must be a Store operation."
        );

        // 4. Process the expression part of the root `Store` operation.
        // The last source of the `Store` UOp is the value to be stored.
        let value_to_store = root.0.src.last().unwrap();
        let processed_value = self.process_node(value_to_store);

        // 5. Append the generated instructions for the loop body.
        linearized_kernel.append(&mut self.kernel_body);

        // 6. Recreate the final `Store` instruction with the processed value.
        // The original index expression used "i", we need to replace it with the new loop_var.
        let dest_buffer = root.0.src[0].clone();
        let original_index = root.0.src[1].clone();
        let new_index = self.replace_loop_var(original_index, &loop_var);

        let final_store = UOp::new(
            Op::Store,
            crate::dtype::DType::Unit,
            vec![dest_buffer, new_index, processed_value],
        );
        linearized_kernel.push(final_store);

        // 7. End the loop with a `LoopEnd` instruction.
        linearized_kernel.push(UOp::new(Op::LoopEnd, crate::dtype::DType::Unit, vec![]));

        debug!("Linearized Kernel: {linearized_kernel:?}");
        linearized_kernel
    }

    /// Replaces instances of the old loop variable "i" with the new one.
    fn replace_loop_var(&self, uop: UOp, new_loop_var: &UOp) -> UOp {
        if let Op::Var(name) = &uop.0.op {
            if name == "i" {
                return new_loop_var.clone();
            }
        }
        let new_srcs = uop
            .0
            .src
            .iter()
            .map(|src| self.replace_loop_var(src.clone(), new_loop_var))
            .collect();
        UOp::new(uop.0.op.clone(), uop.0.dtype.clone(), new_srcs)
    }
}
