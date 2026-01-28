//! Adapter for integrating FusionPass into the GraphSuggester framework
//!
//! This module provides `FusionSuggester<F>`, which wraps a `FusionPass`
//! and exposes it as a `GraphSuggester` for use in beam search optimization.

use crate::graph::{GraphNode, count_nodes};
use crate::lowerer::FusionPass;
use crate::opt::graph::{GraphSuggestResult, GraphSuggester};

/// Adapter that wraps a FusionPass as a GraphSuggester
///
/// This allows fusion passes to participate in beam search optimization
/// alongside other suggesters like MatMulDetector.
pub struct FusionSuggester<F: FusionPass> {
    fusion: F,
    name: String,
}

impl<F: FusionPass> FusionSuggester<F> {
    /// Create a new FusionSuggester wrapping the given FusionPass
    pub fn new(fusion: F, name: impl Into<String>) -> Self {
        Self {
            fusion,
            name: name.into(),
        }
    }
}

impl<F: FusionPass> GraphSuggester for FusionSuggester<F> {
    fn name(&self) -> &str {
        &self.name
    }

    fn suggest(&self, roots: &[GraphNode]) -> Vec<GraphSuggestResult> {
        let fused = self.fusion.apply(roots);

        // Check if the fusion actually changed the graph
        if graphs_differ(roots, &fused) {
            vec![GraphSuggestResult::with_description(
                fused,
                &self.name,
                "fusion applied",
            )]
        } else {
            vec![]
        }
    }
}

/// Check if two graphs are different
///
/// Uses node count as a heuristic - if fusion reduced nodes, something changed.
/// Also checks if root nodes are the same pointers.
fn graphs_differ(old: &[GraphNode], new: &[GraphNode]) -> bool {
    // Different number of roots means definitely different
    if old.len() != new.len() {
        return true;
    }

    // Check if root pointers are different
    for (o, n) in old.iter().zip(new.iter()) {
        if o != n {
            return true;
        }
    }

    // Same root pointers but could have different subgraph
    // Check node counts as a heuristic
    let old_count = count_nodes(old);
    let new_count = count_nodes(new);
    old_count != new_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::{Expr, input};
    use crate::lowerer::ViewFusion;

    #[test]
    fn test_fusion_suggester_no_change() {
        let suggester = FusionSuggester::new(ViewFusion, "view_fusion");

        // Single input with no views to fuse
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = x.unsqueeze(0);

        let suggestions = suggester.suggest(&[y]);
        // No consecutive views, so no fusion possible
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_fusion_suggester_with_change() {
        let suggester = FusionSuggester::new(ViewFusion, "view_fusion");

        // Create consecutive views that can be fused
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let v1 = x.permute(&[1, 0]);
        let v2 = v1.unsqueeze(0);

        let suggestions = suggester.suggest(&[v2]);
        // Consecutive views should produce a fusion suggestion
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].suggester_name, "view_fusion");
    }

    #[test]
    fn test_graphs_differ() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

        // Same node should not differ
        assert!(!graphs_differ(&[x.clone()], &[x.clone()]));

        // Different nodes should differ
        assert!(graphs_differ(&[x.clone()], &[y.clone()]));

        // Different lengths should differ
        assert!(graphs_differ(&[x.clone()], &[x.clone(), y.clone()]));
    }
}
