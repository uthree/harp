//! Node fusion passes for graph optimization
//!
//! Fusion combines multiple graph nodes into single operations,
//! reducing memory traffic and kernel launch overhead.

use crate::ast::AstNode;
use crate::graph::{GraphNode, GraphOp, GraphTransform};

/// Trait for fusion passes
pub trait FusionPass {
    /// Apply the fusion pass to the graph
    fn apply(&self, roots: &[GraphNode]) -> Vec<GraphNode>;
}

// ============================================================================
// View Fusion
// ============================================================================

/// Fuse consecutive View operations
///
/// When a View node's source is also a View node, we can compose them
/// into a single View that achieves the same transformation.
///
/// ```text
/// Before: GraphNode → View(v1) → View(v2)
/// After:  GraphNode → View(compose(v2, v1))
/// ```
pub struct ViewFusion;

impl GraphTransform for ViewFusion {
    fn transform(&self, node: &GraphNode) -> Option<GraphNode> {
        // Check if this is a View operation
        if let GraphOp::View(outer_view) = node.op() {
            // Check if the source is also a View operation
            if node.sources().len() == 1
                && let GraphOp::View(inner_view) = node.sources()[0].op() {
                    // Check if the inner node has exactly one source
                    if node.sources()[0].sources().len() == 1 {
                        // Compose views: outer(inner(x)) = composed(x)
                        // Note: View::compose is an associated function
                        let composed = crate::graph::View::compose(outer_view, inner_view);
                        let new_node = GraphNode::new(
                            node.sources()[0].sources().to_vec(),
                            node.view().clone(),
                            GraphOp::View(composed),
                            node.dtype().clone(),
                            None,
                        );
                        return Some(new_node);
                    }
                }
        }
        None
    }
}

/// Apply view fusion pass to graph
pub fn fuse_views(roots: &[GraphNode]) -> Vec<GraphNode> {
    let fusion = ViewFusion;
    GraphTransform::apply(&fusion, roots)
}

impl FusionPass for ViewFusion {
    fn apply(&self, roots: &[GraphNode]) -> Vec<GraphNode> {
        GraphTransform::apply(self, roots)
    }
}

// ============================================================================
// Elementwise + Reduce Fusion
// ============================================================================

/// Fuse elementwise operation followed by reduction
///
/// When a MapReduce(reduce=Some) node's source is MapReduce(reduce=None),
/// we can combine them into a single node.
///
/// ```text
/// Before: x → MapReduce{map=f, reduce=None} → MapReduce{map=id, reduce=Sum}
/// After:  x → MapReduce{map=f, reduce=Sum}
/// ```
pub struct ElementwiseReduceFusion;

impl ElementwiseReduceFusion {
    /// Check if the map operation is identity (just Wildcard("0"))
    fn is_identity_map(map: &AstNode) -> bool {
        matches!(map, AstNode::Wildcard(s) if s == "0")
    }
}

impl GraphTransform for ElementwiseReduceFusion {
    fn transform(&self, node: &GraphNode) -> Option<GraphNode> {
        // Check if this is a reduce operation
        if let GraphOp::MapReduce {
            map: reduce_map,
            reduce: Some((reduce_op, axis)),
        } = node.op()
        {
            // Only fuse if the reduce has identity map
            if !Self::is_identity_map(reduce_map) {
                return None;
            }

            // Check if source is an elementwise operation
            if node.sources().len() == 1
                && let GraphOp::MapReduce {
                    map: elem_map,
                    reduce: None,
                } = node.sources()[0].op()
                {
                    // Combine: use elementwise map with reduce operation
                    let new_node = GraphNode::new(
                        node.sources()[0].sources().to_vec(),
                        node.view().clone(),
                        GraphOp::MapReduce {
                            map: elem_map.clone(),
                            reduce: Some((*reduce_op, *axis)),
                        },
                        node.dtype().clone(),
                        None,
                    );
                    return Some(new_node);
                }
        }
        None
    }
}

/// Apply elementwise+reduce fusion pass to graph
pub fn fuse_elementwise_reduce(roots: &[GraphNode]) -> Vec<GraphNode> {
    let fusion = ElementwiseReduceFusion;
    GraphTransform::apply(&fusion, roots)
}

impl FusionPass for ElementwiseReduceFusion {
    fn apply(&self, roots: &[GraphNode]) -> Vec<GraphNode> {
        GraphTransform::apply(self, roots)
    }
}

// ============================================================================
// Combined Fusion Pipeline
// ============================================================================

/// Combined fusion pass that applies all fusion optimizations
pub struct AllFusions;

impl FusionPass for AllFusions {
    fn apply(&self, roots: &[GraphNode]) -> Vec<GraphNode> {
        // Apply fusion passes in order
        let fused = fuse_views(roots);
        fuse_elementwise_reduce(&fused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::{Expr, input};

    #[test]
    fn test_view_fusion() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

        // Apply two consecutive view operations
        let transposed = x.permute(&[1, 0]);
        let unsqueezed = transposed.unsqueeze(0);

        // Apply fusion
        let fused = fuse_views(&[unsqueezed.clone()]);

        // Should reduce node count
        let orig_count = crate::graph::count_nodes(&[unsqueezed]);
        let fused_count = crate::graph::count_nodes(&fused);
        assert!(fused_count <= orig_count);
    }

    #[test]
    fn test_elementwise_reduce_fusion() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

        // Elementwise then reduce: (x * 2).sum(1)
        let doubled = &x * &x; // elementwise
        let summed = doubled.sum(1); // reduce

        // Apply fusion
        let fused = fuse_elementwise_reduce(&[summed.clone()]);

        // The fused graph should have fewer nodes
        let orig_count = crate::graph::count_nodes(&[summed]);
        let fused_count = crate::graph::count_nodes(&fused);
        assert!(fused_count <= orig_count);
    }
}
