//! Backward pass implementation for automatic differentiation.

use std::collections::HashSet;

use crate::Tensor;
use crate::autograd::context::GradientContext;
use crate::autograd::vjp::compute_vjp;
use crate::uop::UOp;

/// Performs backward pass (reverse-mode automatic differentiation).
///
/// Computes gradients of the loss with respect to all tensors with requires_grad=true
/// in the computation graph rooted at `output`.
///
/// # Arguments
/// * `output` - The tensor to differentiate (typically a scalar loss)
/// * `grad_output` - Initial gradient (defaults to 1.0 for scalar outputs)
/// * `retain_graph` - Whether to keep the computation graph for multiple backward passes
///
/// # Returns
/// A GradientContext containing gradients for all requires_grad tensors.
pub fn backward(
    output: &Tensor,
    grad_output: Option<Tensor>,
    retain_graph: bool,
    requires_grad_set: &HashSet<usize>,
) -> GradientContext {
    let _ = retain_graph; // Reserved for future use

    let mut ctx = GradientContext::new();

    // Initialize gradient for output
    let grad = grad_output.unwrap_or_else(|| {
        assert!(
            output.shape().is_scalar() || output.numel() == 1,
            "backward() requires grad_output for non-scalar outputs"
        );
        Tensor::full(output.shape().clone(), 1.0f32)
    });

    // Set initial gradient
    ctx.set(output.uop().ptr_id(), grad);

    // Get reverse topological order
    let sorted = topological_sort_reverse(output.uop(), requires_grad_set);

    // Process each node in reverse topological order
    for uop in sorted {
        let ptr_id = uop.ptr_id();

        // Skip if no gradient accumulated
        let current_grad = match ctx.get_by_id(ptr_id) {
            Some(g) => g.clone(),
            None => continue,
        };

        // Compute VJPs for inputs
        let vjps = compute_vjp(&uop, &current_grad);

        // Accumulate gradients for all inputs (they will be used to propagate
        // gradients further down the graph)
        for (input_id, input_grad) in vjps {
            ctx.accumulate(input_id, input_grad);
        }
    }

    ctx
}

/// Returns nodes in reverse topological order (from output to inputs).
///
/// Only includes nodes that are on the path to requires_grad tensors.
fn topological_sort_reverse(root: &UOp, requires_grad_set: &HashSet<usize>) -> Vec<UOp> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    // First, find all nodes reachable from requires_grad nodes
    let mut relevant = HashSet::new();
    mark_relevant(root, requires_grad_set, &mut relevant, &mut HashSet::new());

    // DFS to build topological order
    fn visit(
        uop: &UOp,
        visited: &mut HashSet<usize>,
        order: &mut Vec<UOp>,
        relevant: &HashSet<usize>,
    ) {
        let id = uop.ptr_id();

        if visited.contains(&id) {
            return;
        }

        // Skip if not on path to requires_grad nodes
        if !relevant.contains(&id) {
            return;
        }

        visited.insert(id);

        // Visit children first
        for src in uop.src() {
            visit(src, visited, order, relevant);
        }

        // Add to order (will be reversed)
        order.push(uop.clone());
    }

    visit(root, &mut visited, &mut order, &relevant);

    // Reverse to get order from output to inputs
    order.reverse();
    order
}

/// Marks all nodes on paths from root to requires_grad nodes.
fn mark_relevant(
    uop: &UOp,
    requires_grad_set: &HashSet<usize>,
    relevant: &mut HashSet<usize>,
    visited: &mut HashSet<usize>,
) -> bool {
    let id = uop.ptr_id();

    if visited.contains(&id) {
        return relevant.contains(&id);
    }

    visited.insert(id);

    // Base case: this node requires grad
    let mut is_relevant = requires_grad_set.contains(&id);

    // Recursive case: any child is relevant
    for src in uop.src() {
        if mark_relevant(src, requires_grad_set, relevant, visited) {
            is_relevant = true;
        }
    }

    if is_relevant {
        relevant.insert(id);
    }

    is_relevant
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::ScalarValue;
    use crate::shape::Shape;

    fn make_simple_graph() -> (UOp, HashSet<usize>) {
        // x -> y = x * 2
        let x = UOp::constant(ScalarValue::Float32(3.0), Shape::scalar());
        let two = UOp::constant(ScalarValue::Float32(2.0), Shape::scalar());
        let y = x.mul(&two);

        let mut requires_grad = HashSet::new();
        requires_grad.insert(x.ptr_id());

        (y, requires_grad)
    }

    #[test]
    fn test_topological_sort() {
        let (y, requires_grad) = make_simple_graph();
        let sorted = topological_sort_reverse(&y, &requires_grad);

        // Should have nodes in reverse topological order
        assert!(!sorted.is_empty());
    }
}
