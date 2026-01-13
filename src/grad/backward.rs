//! Backward pass implementation for automatic differentiation
//!
//! This module provides the `backward()` function that computes gradients
//! by propagating through the computation graph in reverse order.

use crate::graph::{GraphNode, ones, topological_sort};

use super::context::{GradContext, GradResult};
use super::rules::compute_vjp;

// ============================================================================
// Backward Function
// ============================================================================

/// Compute gradients for the given parameters with respect to the output.
///
/// This function performs reverse-mode automatic differentiation (backpropagation)
/// to compute `∂output/∂param` for each parameter.
///
/// # Arguments
///
/// * `output` - The scalar output node (typically a loss function)
/// * `params` - The parameter nodes for which to compute gradients
///
/// # Returns
///
/// A `GradResult` containing the gradient for each parameter.
///
/// # Example
///
/// ```ignore
/// use eclat::grad::backward;
/// use eclat::graph::{input, Expr};
///
/// let x = input(vec![Expr::Const(10)], DType::F32);
/// let y = (&x * &x).sum(0);  // y = sum(x^2)
///
/// let grads = backward(&y, &[&x]);
/// let dx = grads.get(&x).unwrap();  // dx = 2*x
/// ```
pub fn backward(output: &GraphNode, params: &[&GraphNode]) -> GradResult {
    let mut ctx = GradContext::new();

    // Mark parameters as requiring gradients
    for param in params {
        ctx.mark_requires_grad(param);
    }

    // Initialize output gradient to ones (assuming scalar loss)
    let initial_grad = ones(output.shape().clone(), output.dtype().clone());
    ctx.set_grad(output, initial_grad);

    // Get topologically sorted nodes (from inputs to output)
    let sorted = topological_sort(std::slice::from_ref(output));

    // Traverse in reverse order (from output to inputs)
    for node in sorted.into_iter().rev() {
        // Skip if this node has no gradient
        let grad_output = match ctx.get_grad(&node) {
            Some(g) => g.clone(),
            None => continue,
        };

        // Skip if this is a leaf node (no sources)
        let sources = node.sources();
        if sources.is_empty() {
            continue;
        }

        // Compute VJP for this node
        let vjp = compute_vjp(&node, &grad_output);

        // Accumulate gradients to source nodes
        for (source, grad_input) in sources.iter().zip(vjp.input_grads) {
            if let Some(grad) = grad_input {
                // Only accumulate if the source is on the path to a parameter
                if needs_grad(&ctx, source) {
                    ctx.accumulate_grad(source, grad);
                }
            }
        }
    }

    // Collect results
    let param_nodes: Vec<GraphNode> = params.iter().map(|p| (*p).clone()).collect();
    GradResult::new(ctx, param_nodes)
}

/// Check if a node needs gradient computation.
///
/// A node needs gradient if:
/// 1. It's a parameter (requires_grad is true), or
/// 2. Any of its descendants is a parameter
fn needs_grad(ctx: &GradContext, node: &GraphNode) -> bool {
    if ctx.requires_grad(node) {
        return true;
    }

    // Check descendants (this is inefficient but simple)
    // A better implementation would cache this during forward pass
    for source in node.sources() {
        if needs_grad(ctx, source) {
            return true;
        }
    }

    false
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Compute gradient of output with respect to a single parameter.
///
/// This is a convenience wrapper around `backward()` for the common case
/// of computing gradient for a single parameter.
pub fn grad(output: &GraphNode, param: &GraphNode) -> Option<GraphNode> {
    let result = backward(output, &[param]);
    result.get(param)
}

/// Compute gradients of output with respect to multiple parameters.
///
/// Returns gradients in the same order as the input parameters.
pub fn grads(output: &GraphNode, params: &[&GraphNode]) -> Vec<Option<GraphNode>> {
    let result = backward(output, params);
    result.grads()
}

// ============================================================================
// Extension Trait for GraphNode
// ============================================================================

/// Extension trait to add backward() method to GraphNode.
pub trait Differentiable {
    /// Compute gradients with respect to the given parameters.
    fn backward(&self, params: &[&GraphNode]) -> GradResult;

    /// Compute gradient with respect to a single parameter.
    fn grad(&self, param: &GraphNode) -> Option<GraphNode>;
}

impl Differentiable for GraphNode {
    fn backward(&self, params: &[&GraphNode]) -> GradResult {
        backward(self, params)
    }

    fn grad(&self, param: &GraphNode) -> Option<GraphNode> {
        grad(self, param)
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
    fn test_backward_simple_add() {
        // y = a + b
        // dy/da = 1, dy/db = 1
        let a = input(vec![Expr::Const(10)], DType::F32);
        let b = input(vec![Expr::Const(10)], DType::F32);
        let y = &a + &b;

        let grads = backward(&y, &[&a, &b]);

        let da = grads.get(&a).expect("Should have gradient for a");
        let db = grads.get(&b).expect("Should have gradient for b");

        // Gradients should have same shape as inputs
        assert_eq!(da.shape(), a.shape());
        assert_eq!(db.shape(), b.shape());
    }

    #[test]
    fn test_backward_mul() {
        // y = a * b
        // dy/da = b, dy/db = a
        let a = input(vec![Expr::Const(10)], DType::F32);
        let b = input(vec![Expr::Const(10)], DType::F32);
        let y = &a * &b;

        let grads = backward(&y, &[&a, &b]);

        let da = grads.get(&a).expect("Should have gradient for a");
        let db = grads.get(&b).expect("Should have gradient for b");

        assert_eq!(da.shape(), a.shape());
        assert_eq!(db.shape(), b.shape());
    }

    #[test]
    fn test_backward_chain() {
        // y = (a + b) * (a - b)
        // y = a^2 - b^2
        // dy/da = 2a, dy/db = -2b
        let a = input(vec![Expr::Const(10)], DType::F32);
        let b = input(vec![Expr::Const(10)], DType::F32);
        let sum = &a + &b;
        let diff = &a - &b;
        let y = &sum * &diff;

        let grads = backward(&y, &[&a, &b]);

        let da = grads.get(&a).expect("Should have gradient for a");
        let db = grads.get(&b).expect("Should have gradient for b");

        assert_eq!(da.shape(), a.shape());
        assert_eq!(db.shape(), b.shape());
    }

    #[test]
    fn test_backward_sum() {
        // y = sum(a)
        // dy/da = ones
        let a = input(vec![Expr::Const(10), Expr::Const(20)], DType::F32);
        let y = a.sum(1).sum(0);

        let grads = backward(&y, &[&a]);
        let da = grads.get(&a).expect("Should have gradient for a");

        // Gradient should be expanded back to original shape
        assert_eq!(da.shape(), a.shape());
    }

    #[test]
    fn test_backward_neg() {
        // y = -a
        // dy/da = -1
        let a = input(vec![Expr::Const(10)], DType::F32);
        let y = -&a;

        let grads = backward(&y, &[&a]);
        let da = grads.get(&a).expect("Should have gradient for a");

        assert_eq!(da.shape(), a.shape());
    }

    #[test]
    fn test_grad_convenience() {
        let a = input(vec![Expr::Const(10)], DType::F32);
        let y = &a * &a; // y = a^2

        let da = y.grad(&a).expect("Should have gradient");
        assert_eq!(da.shape(), a.shape());
    }

    #[test]
    fn test_backward_reused_node() {
        // y = a * a (a is used twice)
        // dy/da = a + a = 2a
        let a = input(vec![Expr::Const(10)], DType::F32);
        let y = &a * &a;

        let grads = backward(&y, &[&a]);
        let da = grads.get(&a).expect("Should have gradient for a");

        // Gradient should be accumulated
        assert_eq!(da.shape(), a.shape());
    }

    #[test]
    fn test_backward_no_grad_for_non_param() {
        let a = input(vec![Expr::Const(10)], DType::F32);
        let b = input(vec![Expr::Const(10)], DType::F32);
        let y = &a + &b;

        // Only ask for gradient of a
        let grads = backward(&y, &[&a]);

        assert!(grads.get(&a).is_some());
        // b was not requested, so shouldn't be in results
        // (but get() will return None which is fine)
    }
}
