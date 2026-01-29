//! View fusion suggester
//!
//! Fuses consecutive View operations into a single View.

use crate::graph::{GraphNode, count_nodes};
use crate::lowerer::fuse_views;
use crate::opt::graph::{GraphSuggestResult, GraphSuggester};

/// Fuse consecutive View operations
///
/// When a View node's source is also a View node, we can compose them
/// into a single View that achieves the same transformation.
///
/// ```text
/// Before: GraphNode → View(v1) → View(v2)
/// After:  GraphNode → View(compose(v2, v1))
/// ```
pub struct ViewFusionSuggester;

impl ViewFusionSuggester {
    /// Create a new ViewFusionSuggester
    pub fn new() -> Self {
        Self
    }
}

impl Default for ViewFusionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ViewFusionSuggester {
    fn name(&self) -> &str {
        "ViewFusion"
    }

    fn suggest(&self, roots: &[GraphNode]) -> Vec<GraphSuggestResult> {
        let fused = fuse_views(roots);

        // Check if the fusion actually changed the graph
        if graphs_differ(roots, &fused) {
            vec![GraphSuggestResult::with_description(
                fused,
                self.name(),
                "consecutive views fused",
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
    fn test_no_change() {
        let suggester = ViewFusionSuggester::new();

        // Single view - nothing to fuse
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = x.unsqueeze(0);

        let suggestions = suggester.suggest(&[y]);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_with_change() {
        let suggester = ViewFusionSuggester::new();

        // Consecutive views that can be fused
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let v1 = x.permute(&[1, 0]);
        let v2 = v1.unsqueeze(0);

        let suggestions = suggester.suggest(&[v2]);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].suggester_name, "ViewFusion");
    }
}
