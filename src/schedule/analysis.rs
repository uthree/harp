//! Graph analysis for fusion decisions.

use std::collections::{HashMap, HashSet};

use crate::ops::Ops;
use crate::uop::UOp;

/// Graph analysis helper for making fusion decisions.
pub struct GraphAnalysis {
    /// Map from UOp pointer ID to reference count
    ref_counts: HashMap<usize, usize>,
    /// Set of UOp pointer IDs that have been visited
    visited: HashSet<usize>,
}

impl GraphAnalysis {
    /// Create a new graph analysis for the given output UOp.
    /// This traverses the graph and counts references.
    pub fn new(output: &UOp) -> Self {
        let mut analysis = Self {
            ref_counts: HashMap::new(),
            visited: HashSet::new(),
        };
        analysis.count_refs(output);
        analysis
    }

    /// Count references for all nodes in the graph.
    fn count_refs(&mut self, uop: &UOp) {
        let id = uop.ptr_id();
        if self.visited.contains(&id) {
            return;
        }
        self.visited.insert(id);

        for src in uop.src() {
            let src_id = src.ptr_id();
            *self.ref_counts.entry(src_id).or_insert(0) += 1;
            self.count_refs(src);
        }
    }

    /// Get the reference count for a UOp.
    pub fn ref_count(&self, uop: &UOp) -> usize {
        self.ref_counts.get(&uop.ptr_id()).copied().unwrap_or(0)
    }

    /// Check if a UOp has exactly one consumer (can be fused into its consumer).
    pub fn has_single_consumer(&self, uop: &UOp) -> bool {
        self.ref_count(uop) == 1
    }

    /// Check if two UOps can be fused together (elementwise fusion).
    ///
    /// Conditions for elementwise fusion:
    /// 1. Both operations are elementwise
    /// 2. Shapes match
    /// 3. The source has only one consumer (the current op)
    /// 4. The source is not a Load operation (need buffer input)
    pub fn can_fuse_elementwise(&self, consumer: &UOp, source: &UOp) -> bool {
        // Check if consumer is elementwise
        if !consumer.op().is_elementwise() && !consumer.op().is_compare() {
            return false;
        }

        // Check if source is elementwise (or compare, which is also fusable)
        let source_op = source.op();
        if !source_op.is_elementwise() && !source_op.is_compare() {
            return false;
        }

        // Don't fuse Load operations - they provide input buffers
        if matches!(source_op, Ops::Load | Ops::Const) {
            return false;
        }

        // Check shapes match
        if consumer.shape() != source.shape() {
            return false;
        }

        // Source must have only one consumer (this consumer)
        if !self.has_single_consumer(source) {
            return false;
        }

        true
    }

    /// Check if a reduce operation can fuse its elementwise input.
    pub fn can_fuse_reduce_input(&self, reduce: &UOp, source: &UOp) -> bool {
        // Reduce must be a reduce operation
        if !reduce.op().is_reduce() {
            return false;
        }

        // Source must be elementwise
        let source_op = source.op();
        if !source_op.is_elementwise() && !source_op.is_compare() {
            return false;
        }

        // Don't fuse Load/Const
        if matches!(source_op, Ops::Load | Ops::Const) {
            return false;
        }

        // Source must have only one consumer
        self.has_single_consumer(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::shape::Shape;

    #[test]
    fn test_ref_counting() {
        // Create a simple graph: a + a (a is referenced twice)
        let a = UOp::load(0, DType::Float32, Shape::from(vec![4]));
        let sum = a.add(&a);

        let analysis = GraphAnalysis::new(&sum);
        assert_eq!(analysis.ref_count(&a), 2);
    }

    #[test]
    fn test_single_consumer() {
        // Create: neg(a) + b
        let a = UOp::load(0, DType::Float32, Shape::from(vec![4]));
        let b = UOp::load(1, DType::Float32, Shape::from(vec![4]));
        let neg_a = a.neg();
        let sum = neg_a.add(&b);

        let analysis = GraphAnalysis::new(&sum);
        assert!(analysis.has_single_consumer(&neg_a));
        assert!(analysis.has_single_consumer(&a));
        assert!(analysis.has_single_consumer(&b));
    }

    #[test]
    fn test_elementwise_fusion_candidate() {
        let a = UOp::load(0, DType::Float32, Shape::from(vec![4]));
        let b = UOp::load(1, DType::Float32, Shape::from(vec![4]));
        let neg_a = a.neg();
        let sum = neg_a.add(&b);

        let analysis = GraphAnalysis::new(&sum);

        // sum can fuse neg_a
        assert!(analysis.can_fuse_elementwise(&sum, &neg_a));

        // sum cannot fuse b (it's a Load)
        assert!(!analysis.can_fuse_elementwise(&sum, &b));
    }
}
