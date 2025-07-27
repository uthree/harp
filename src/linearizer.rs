//! # `UOp` Linearizer
//!
//! This module implements the linearization process, which transforms a `UOp`
//! computation graph into a flat, sequential list of `UOp` instructions. This
//! linear representation is ideal for code generation, as the renderer can simply
//! iterate through the list and emit the corresponding code for each instruction.
//!
//! The linearizer handles control flow, such as loops, by introducing special
//! `Op::LoopStart` and `Op::LoopEnd` instructions into the instruction stream.
//! It also performs kernel fusion by inlining `UOp`s that are only used once,
//! reducing redundant memory access.

use crate::uop::{Op, UOp, UOp_};
use log::debug;
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// A structure that holds the state for the linearization process.
pub struct Linearizer<'a> {
    node_map: FxHashMap<*const UOp_, UOp>,
    kernel_body: Vec<UOp>,
    use_counts: &'a FxHashMap<*const UOp_, usize>,
    var_counter: usize,
    loop_var_counter: usize,
}

impl<'a> Linearizer<'a> {
    /// Creates a new `Linearizer`.
    pub fn new(use_counts: &'a FxHashMap<*const UOp_, usize>) -> Self {
        Self {
            node_map: FxHashMap::default(),
            kernel_body: Vec::new(),
            use_counts,
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
    /// This method recursively traverses the graph. If a node is used more than
    /// once, its result is stored in a temporary variable. If it's used only
    /// once, it's inlined directly into its parent's expression (fusion).
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
            // These ops are never stored in variables, they are returned directly.
            Op::Const(_) | Op::Var(_) => {
                return UOp::new(node.0.op.clone(), node.0.dtype.clone(), new_srcs);
            }
            _ => {
                // Check the use count of the current node.
                let use_count = self.use_counts.get(&node_ptr).copied().unwrap_or(0);

                // If used more than once, or if it's a Load (must be a variable),
                // store the result in a temporary variable.
                if use_count > 1 || matches!(node.0.op, Op::Load) {
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
                } else {
                    // Fusion: If used once, inline the expression directly.
                    UOp::new(node.0.op.clone(), node.0.dtype.clone(), new_srcs)
                }
            }
        };

        self.node_map.insert(node_ptr, result_uop.clone());
        result_uop
    }

    /// Linearizes the entire computation graph starting from a root `UOp`.
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
        let value_to_store = root.0.src.last().unwrap();
        let value_to_store = self.replace_loop_var(value_to_store.clone(), &loop_var);
        let processed_value = self.process_node(&value_to_store);

        // 5. Append the generated instructions for the loop body.
        linearized_kernel.append(&mut self.kernel_body);

        // 6. Recreate the final `Store` instruction with the processed value.
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
