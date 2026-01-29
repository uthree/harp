//! Elementwise + Reduce fusion suggester
//!
//! Fuses elementwise operations with subsequent reductions.

use crate::graph::{GraphNode, count_nodes};
use crate::lowerer::fuse_elementwise_reduce;
use crate::opt::graph::{GraphSuggestResult, GraphSuggester};

/// Fuse elementwise operation followed by reduction
///
/// When a MapReduce(reduce=Some) node's source is MapReduce(reduce=None),
/// we can combine them into a single node.
///
/// ```text
/// Before: x → MapReduce{map=f, reduce=None} → MapReduce{map=id, reduce=Sum}
/// After:  x → MapReduce{map=f, reduce=Sum}
/// ```
pub struct ElementwiseReduceFusionSuggester;

impl ElementwiseReduceFusionSuggester {
    /// Create a new ElementwiseReduceFusionSuggester
    pub fn new() -> Self {
        Self
    }
}

impl Default for ElementwiseReduceFusionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ElementwiseReduceFusionSuggester {
    fn name(&self) -> &str {
        "ElementwiseReduceFusion"
    }

    fn suggest(&self, roots: &[GraphNode]) -> Vec<GraphSuggestResult> {
        let fused = fuse_elementwise_reduce(roots);

        // Check if the fusion actually changed the graph
        if graphs_differ(roots, &fused) {
            vec![GraphSuggestResult::with_description(
                fused,
                self.name(),
                "elementwise and reduce fused",
            )]
        } else {
            vec![]
        }
    }
}

/// Check if two graphs are different
fn graphs_differ(old: &[GraphNode], new: &[GraphNode]) -> bool {
    if old.len() != new.len() {
        return true;
    }

    for (o, n) in old.iter().zip(new.iter()) {
        if o != n {
            return true;
        }
    }

    let old_count = count_nodes(old);
    let new_count = count_nodes(new);
    old_count != new_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::{Expr, input};

    #[test]
    fn test_elementwise_reduce_fusion() {
        let suggester = ElementwiseReduceFusionSuggester::new();

        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let doubled = &x * &x; // elementwise
        let summed = doubled.sum(1); // reduce

        let suggestions = suggester.suggest(&[summed]);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].suggester_name, "ElementwiseReduceFusion");
    }

    #[test]
    fn test_no_fusion_when_no_elementwise() {
        let suggester = ElementwiseReduceFusionSuggester::new();

        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let summed = x.sum(1); // just reduce, no elementwise

        let suggestions = suggester.suggest(&[summed]);
        // Input → reduce, no elementwise to fuse
        assert!(suggestions.is_empty());
    }
}
