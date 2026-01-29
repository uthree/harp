//! Scheduler for creating optimized kernel execution plans.

use std::collections::{HashMap, HashSet};

use crate::ops::Ops;
use crate::uop::{UOp, UOpArg};

use super::analysis::GraphAnalysis;
use super::item::ScheduleItem;
use super::kernel::{FusedKernel, FusedOp, FusedSource, KernelInput};

/// Scheduler creates optimized execution schedules by fusing operations.
pub struct Scheduler {
    analysis: GraphAnalysis,
    /// Map from UOp pointer ID to buffer index (for inputs)
    buffer_map: HashMap<usize, usize>,
    /// Next buffer index to assign
    next_buffer_idx: usize,
    /// Set of UOp pointer IDs that have been scheduled
    scheduled: HashSet<usize>,
}

impl Scheduler {
    /// Create a new scheduler for the given output UOp.
    pub fn new(output: &UOp) -> Self {
        Self {
            analysis: GraphAnalysis::new(output),
            buffer_map: HashMap::new(),
            next_buffer_idx: 0,
            scheduled: HashSet::new(),
        }
    }

    /// Create a schedule for evaluating the given UOp.
    /// Returns a list of schedule items to execute in order.
    pub fn schedule(&mut self, output: &UOp) -> Vec<ScheduleItem> {
        let mut items = Vec::new();
        self.schedule_recursive(output, &mut items);
        items
    }

    fn schedule_recursive(&mut self, uop: &UOp, items: &mut Vec<ScheduleItem>) {
        let id = uop.ptr_id();
        if self.scheduled.contains(&id) {
            return;
        }

        match uop.op() {
            Ops::Load => {
                // Load operations assign buffer indices but don't create schedule items
                self.get_or_assign_buffer(uop);
                self.scheduled.insert(id);
            }
            Ops::Const => {
                // Constants are inlined, no schedule item needed
                self.scheduled.insert(id);
            }
            op if op.is_elementwise() || op.is_compare() => {
                self.schedule_elementwise(uop, items);
            }
            op if op.is_reduce() => {
                self.schedule_reduce(uop, items);
            }
            op if op.is_movement() => {
                self.schedule_movement(uop, items);
            }
            Ops::Cast => {
                self.schedule_cast(uop, items);
            }
            Ops::Where => {
                self.schedule_where(uop, items);
            }
            _ => {
                // Fallback: schedule sources first, then single op
                for src in uop.src() {
                    self.schedule_recursive(src, items);
                }
                let inputs = self.collect_input_indices(uop);
                items.push(ScheduleItem::single(uop.clone(), inputs));
                self.scheduled.insert(id);
            }
        }
    }

    fn schedule_elementwise(&mut self, uop: &UOp, items: &mut Vec<ScheduleItem>) {
        let id = uop.ptr_id();
        if self.scheduled.contains(&id) {
            return;
        }

        // Collect fusable chain
        let (fused_ops, leaf_inputs) = self.collect_elementwise_chain(uop);

        // Schedule leaf inputs first
        for leaf in &leaf_inputs {
            self.schedule_recursive(leaf, items);
        }

        // Collect buffer indices for inputs
        let inputs: Vec<usize> = leaf_inputs
            .iter()
            .filter_map(|u| self.buffer_map.get(&u.ptr_id()).copied())
            .collect();

        // Create schedule item
        if fused_ops.len() > 1 {
            items.push(ScheduleItem::elementwise(uop.clone(), fused_ops, inputs));
        } else {
            items.push(ScheduleItem::single(uop.clone(), inputs));
        }

        self.scheduled.insert(id);
    }

    fn schedule_reduce(&mut self, uop: &UOp, items: &mut Vec<ScheduleItem>) {
        let id = uop.ptr_id();
        if self.scheduled.contains(&id) {
            return;
        }

        let sources = uop.src();
        if sources.is_empty() {
            self.scheduled.insert(id);
            return;
        }

        let source = &sources[0];

        // Check if we can fuse elementwise ops before reduce
        if self.analysis.can_fuse_reduce_input(uop, source) {
            // Collect elementwise chain before reduce
            let (mut fused_ops, leaf_inputs) = self.collect_elementwise_chain(source);

            // Schedule leaf inputs
            for leaf in &leaf_inputs {
                self.schedule_recursive(leaf, items);
            }

            // Add the reduce operation
            fused_ops.push(uop.clone());

            let inputs: Vec<usize> = leaf_inputs
                .iter()
                .filter_map(|u| self.buffer_map.get(&u.ptr_id()).copied())
                .collect();

            items.push(ScheduleItem::reduce(uop.clone(), fused_ops, inputs));
        } else {
            // No fusion, schedule source first
            self.schedule_recursive(source, items);
            let inputs = self.collect_input_indices(uop);
            items.push(ScheduleItem::single(uop.clone(), inputs));
        }

        self.scheduled.insert(id);
    }

    fn schedule_movement(&mut self, uop: &UOp, items: &mut Vec<ScheduleItem>) {
        let id = uop.ptr_id();
        if self.scheduled.contains(&id) {
            return;
        }

        // Schedule sources
        for src in uop.src() {
            self.schedule_recursive(src, items);
        }

        // Movement ops are not fused for now
        let inputs = self.collect_input_indices(uop);
        items.push(ScheduleItem::single(uop.clone(), inputs));
        self.scheduled.insert(id);
    }

    fn schedule_cast(&mut self, uop: &UOp, items: &mut Vec<ScheduleItem>) {
        let id = uop.ptr_id();
        if self.scheduled.contains(&id) {
            return;
        }

        for src in uop.src() {
            self.schedule_recursive(src, items);
        }

        let inputs = self.collect_input_indices(uop);
        items.push(ScheduleItem::single(uop.clone(), inputs));
        self.scheduled.insert(id);
    }

    fn schedule_where(&mut self, uop: &UOp, items: &mut Vec<ScheduleItem>) {
        let id = uop.ptr_id();
        if self.scheduled.contains(&id) {
            return;
        }

        // Schedule all three sources
        for src in uop.src() {
            self.schedule_recursive(src, items);
        }

        let inputs = self.collect_input_indices(uop);
        items.push(ScheduleItem::single(uop.clone(), inputs));
        self.scheduled.insert(id);
    }

    /// Collect a chain of fusable elementwise operations.
    /// Returns (ops in execution order, leaf inputs).
    fn collect_elementwise_chain(&self, uop: &UOp) -> (Vec<UOp>, Vec<UOp>) {
        let mut ops = Vec::new();
        let mut leaves = Vec::new();
        let mut visited = HashSet::new();

        self.collect_chain_recursive(uop, &mut ops, &mut leaves, &mut visited);

        // Reverse ops to get execution order (leaves first, output last)
        ops.reverse();
        (ops, leaves)
    }

    fn collect_chain_recursive(
        &self,
        uop: &UOp,
        ops: &mut Vec<UOp>,
        leaves: &mut Vec<UOp>,
        visited: &mut HashSet<usize>,
    ) {
        let id = uop.ptr_id();
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        let op = uop.op();

        // Check if this is a leaf (Load, Const, or non-fusable)
        if matches!(op, Ops::Load | Ops::Const) {
            leaves.push(uop.clone());
            return;
        }

        if !op.is_elementwise() && !op.is_compare() {
            // Non-fusable op is a leaf
            leaves.push(uop.clone());
            return;
        }

        // Add this op to the chain
        ops.push(uop.clone());

        // For each source, check if it can be fused
        for src in uop.src() {
            if self.analysis.can_fuse_elementwise(uop, src) {
                // This source can be fused, recurse into it
                self.collect_chain_recursive(src, ops, leaves, visited);
            } else {
                // This source cannot be fused, treat it as a leaf
                if !visited.contains(&src.ptr_id()) {
                    leaves.push(src.clone());
                    visited.insert(src.ptr_id());
                }
            }
        }
    }

    fn collect_input_indices(&mut self, uop: &UOp) -> Vec<usize> {
        let mut indices = Vec::new();
        for src in uop.src() {
            if let Some(&idx) = self.buffer_map.get(&src.ptr_id()) {
                indices.push(idx);
            }
        }
        indices
    }

    fn get_or_assign_buffer(&mut self, uop: &UOp) -> usize {
        let id = uop.ptr_id();
        if let Some(&idx) = self.buffer_map.get(&id) {
            idx
        } else {
            let idx = self.next_buffer_idx;
            self.buffer_map.insert(id, idx);
            self.next_buffer_idx += 1;
            idx
        }
    }

    /// Assign a buffer index to a UOp (for output buffers).
    pub fn assign_output_buffer(&mut self, uop: &UOp) -> usize {
        self.get_or_assign_buffer(uop)
    }

    /// Build a FusedKernel from a schedule item.
    pub fn build_fused_kernel(&self, item: &ScheduleItem) -> FusedKernel {
        let mut ops_chain = Vec::new();
        let mut inputs = Vec::new();
        let mut op_indices: HashMap<usize, usize> = HashMap::new();

        // Map leaf inputs to kernel inputs
        let mut input_map: HashMap<usize, usize> = HashMap::new();

        // First pass: identify all inputs needed
        for (i, uop) in item.fused_ops.iter().enumerate() {
            for src in uop.src() {
                let src_id = src.ptr_id();

                // Check if it's an already-processed op
                if op_indices.contains_key(&src_id) {
                    continue;
                }

                // Check if it's a constant
                if matches!(src.op(), Ops::Const) {
                    continue;
                }

                // It's an input buffer
                if let std::collections::hash_map::Entry::Vacant(e) = input_map.entry(src_id) {
                    let input_idx = inputs.len();
                    e.insert(input_idx);
                    inputs.push(KernelInput::new(
                        self.buffer_map.get(&src_id).copied().unwrap_or(input_idx),
                        src.dtype(),
                        src.shape().clone(),
                    ));
                }
            }

            op_indices.insert(uop.ptr_id(), i);
        }

        // Second pass: build ops chain with correct sources
        for uop in &item.fused_ops {
            let mut sources = Vec::new();

            for src in uop.src() {
                let src_id = src.ptr_id();

                if let Some(&prev_op_idx) = op_indices.get(&src_id) {
                    // Reference to a previous op in the chain
                    sources.push(FusedSource::PrevOp(prev_op_idx));
                } else if let Some(&input_idx) = input_map.get(&src_id) {
                    // Reference to an input buffer
                    sources.push(FusedSource::Input(input_idx));
                } else if matches!(src.op(), Ops::Const) {
                    // Inline constant
                    if let Some(UOpArg::Scalar(val)) = src.arg() {
                        sources.push(FusedSource::Constant(*val));
                    }
                }
            }

            ops_chain.push(FusedOp::new(uop.op(), sources, uop.dtype()));
        }

        // Generate a name based on operations
        let name = self.generate_kernel_name(&item.fused_ops);

        FusedKernel::new(
            name,
            ops_chain,
            inputs,
            item.output.shape().clone(),
            item.output.dtype(),
        )
    }

    fn generate_kernel_name(&self, ops: &[UOp]) -> String {
        if ops.len() == 1 {
            format!("kernel_{:?}", ops[0].op()).to_lowercase()
        } else {
            let op_names: Vec<String> = ops.iter().map(|u| format!("{:?}", u.op())).collect();
            format!("fused_{}", op_names.join("_")).to_lowercase()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::schedule::FusionType;
    use crate::shape::Shape;

    #[test]
    fn test_simple_schedule() {
        let a = UOp::load(0, DType::Float32, Shape::from(vec![4]));
        let b = UOp::load(1, DType::Float32, Shape::from(vec![4]));
        let sum = a.add(&b);

        let mut scheduler = Scheduler::new(&sum);
        let items = scheduler.schedule(&sum);

        // Should have 1 item for the add
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].fusion_type, FusionType::Single);
    }

    #[test]
    fn test_elementwise_fusion() {
        let a = UOp::load(0, DType::Float32, Shape::from(vec![4]));
        let b = UOp::load(1, DType::Float32, Shape::from(vec![4]));
        let c = UOp::load(2, DType::Float32, Shape::from(vec![4]));

        // (a + b) * c - should fuse add and mul
        let sum = a.add(&b);
        let product = sum.mul(&c);

        let mut scheduler = Scheduler::new(&product);
        let items = scheduler.schedule(&product);

        // Should have 1 fused item
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].fusion_type, FusionType::Elementwise);
        assert_eq!(items[0].fused_ops.len(), 2);
    }

    #[test]
    fn test_build_fused_kernel() {
        let a = UOp::load(0, DType::Float32, Shape::from(vec![4]));
        let b = UOp::load(1, DType::Float32, Shape::from(vec![4]));
        let sum = a.add(&b);
        let neg = sum.neg();

        let mut scheduler = Scheduler::new(&neg);
        let items = scheduler.schedule(&neg);

        assert_eq!(items.len(), 1);

        let kernel = scheduler.build_fused_kernel(&items[0]);
        assert_eq!(kernel.inputs.len(), 2);
        assert_eq!(kernel.ops_chain.len(), 2);
        assert!(kernel.name.contains("fused"));
    }
}
