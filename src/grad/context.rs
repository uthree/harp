//! Gradient computation context
//!
//! This module provides the `GradContext` structure that tracks gradients
//! during backpropagation.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::graph::{GraphInner, GraphNode};

// ============================================================================
// GradContext
// ============================================================================

/// Context for gradient computation during backpropagation.
///
/// `GradContext` maintains a mapping from forward-pass nodes to their gradients,
/// enabling reverse-mode automatic differentiation.
///
/// # How it works
///
/// 1. During forward pass, operations are recorded in the computation graph
/// 2. When `backward()` is called, `GradContext` is created
/// 3. Gradients are propagated from output to inputs via the graph structure
/// 4. Each node's gradient is accumulated (for nodes used multiple times)
#[derive(Default)]
pub struct GradContext {
    /// Mapping from forward node (by pointer) to its gradient node.
    pub(crate) grad_map: HashMap<NodeId, GraphNode>,

    /// Set of nodes that require gradients.
    pub(crate) requires_grad_set: HashSet<NodeId>,
}

/// Node identifier (using pointer for identity).
pub type NodeId = *const GraphInner;

impl GradContext {
    /// Create a new gradient context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a node as requiring gradients.
    pub fn mark_requires_grad(&mut self, node: &GraphNode) {
        self.requires_grad_set.insert(node_id(node));
    }

    /// Check if a node requires gradients.
    pub fn requires_grad(&self, node: &GraphNode) -> bool {
        self.requires_grad_set.contains(&node_id(node))
    }

    /// Get the gradient for a node, if it exists.
    pub fn get_grad(&self, node: &GraphNode) -> Option<&GraphNode> {
        self.grad_map.get(&node_id(node))
    }

    /// Set the gradient for a node.
    pub fn set_grad(&mut self, node: &GraphNode, grad: GraphNode) {
        self.grad_map.insert(node_id(node), grad);
    }

    /// Accumulate gradient for a node.
    ///
    /// If the node already has a gradient, adds to it.
    /// Otherwise, sets the gradient.
    pub fn accumulate_grad(&mut self, node: &GraphNode, grad: GraphNode) {
        let id = node_id(node);

        if let Some(existing) = self.grad_map.get(&id) {
            // Add to existing gradient
            let accumulated = existing + &grad;
            self.grad_map.insert(id, accumulated);
        } else {
            self.grad_map.insert(id, grad);
        }
    }

    /// Check if we should propagate gradients through a node.
    ///
    /// A node should have gradients propagated if:
    /// - It has a gradient set, AND
    /// - Any of its sources require gradients
    pub fn should_propagate(&self, node: &GraphNode) -> bool {
        if !self.grad_map.contains_key(&node_id(node)) {
            return false;
        }

        // Check if any source needs gradients
        for source in node.sources() {
            if self.requires_grad_recursively(source) {
                return true;
            }
        }

        false
    }

    /// Check if a node or any of its ancestors require gradients.
    fn requires_grad_recursively(&self, node: &GraphNode) -> bool {
        if self.requires_grad(node) {
            return true;
        }

        for source in node.sources() {
            if self.requires_grad_recursively(source) {
                return true;
            }
        }

        false
    }

    /// Get all nodes that have gradients computed.
    pub fn grad_nodes(&self) -> impl Iterator<Item = &GraphNode> {
        self.grad_map.values()
    }

    /// Get the number of nodes with computed gradients.
    pub fn num_grads(&self) -> usize {
        self.grad_map.len()
    }
}

/// Get the node ID (pointer) for a GraphNode.
pub fn node_id(node: &GraphNode) -> NodeId {
    Rc::as_ptr(&node.0)
}

// ============================================================================
// GradResult
// ============================================================================

/// Result of backward pass computation.
///
/// Contains the gradients for the requested parameters.
pub struct GradResult {
    /// The gradient context containing all computed gradients.
    context: GradContext,

    /// The parameter nodes for which gradients were requested.
    params: Vec<GraphNode>,
}

impl GradResult {
    /// Create a new GradResult.
    pub(crate) fn new(context: GradContext, params: Vec<GraphNode>) -> Self {
        Self { context, params }
    }

    /// Get the gradient for a specific parameter.
    ///
    /// Returns `None` if the parameter was not in the requested list
    /// or if it has no gradient (e.g., it's not connected to the output).
    pub fn get(&self, param: &GraphNode) -> Option<GraphNode> {
        self.context.get_grad(param).cloned()
    }

    /// Get all computed gradients as a vector.
    ///
    /// Returns gradients in the same order as the parameters were provided.
    /// Returns `None` for parameters without gradients.
    pub fn grads(&self) -> Vec<Option<GraphNode>> {
        self.params
            .iter()
            .map(|p| self.context.get_grad(p).cloned())
            .collect()
    }

    /// Get the underlying gradient context.
    pub fn context(&self) -> &GradContext {
        &self.context
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::{Expr, input};

    #[test]
    fn test_grad_context_basic() {
        let mut ctx = GradContext::new();

        let x = input(vec![Expr::Const(10)], DType::F32);
        let y = input(vec![Expr::Const(10)], DType::F32);

        // Mark x as requiring grad
        ctx.mark_requires_grad(&x);
        assert!(ctx.requires_grad(&x));
        assert!(!ctx.requires_grad(&y));
    }

    #[test]
    fn test_grad_accumulation() {
        let mut ctx = GradContext::new();

        let x = input(vec![Expr::Const(10)], DType::F32);
        let grad1 = input(vec![Expr::Const(10)], DType::F32);
        let grad2 = input(vec![Expr::Const(10)], DType::F32);

        // Accumulate gradients
        ctx.accumulate_grad(&x, grad1.clone());
        ctx.accumulate_grad(&x, grad2.clone());

        let result = ctx.get_grad(&x).unwrap();
        // The result should be grad1 + grad2
        assert_eq!(result.shape(), vec![Expr::Const(10)]);
    }

    #[test]
    fn test_grad_result() {
        let ctx = GradContext::new();
        let x = input(vec![Expr::Const(10)], DType::F32);

        let result = GradResult::new(ctx, vec![x.clone()]);

        // x has no gradient yet
        assert!(result.get(&x).is_none());
    }
}
