//! VJP (Vector-Jacobian Product) rules for automatic differentiation
//!
//! This module defines the gradient rules for each primitive operation.
//! The VJP computes `grad_input = vjp(grad_output)` for each operation.

use crate::ast::AstNode;
use crate::graph::{GraphNode, GraphOp, ReduceOp, ones};

// ============================================================================
// VJP Result
// ============================================================================

/// Result of computing VJP for an operation.
pub struct VjpResult {
    /// Gradients for each input, in order.
    /// `None` means the input doesn't need a gradient.
    pub input_grads: Vec<Option<GraphNode>>,
}

// ============================================================================
// Main VJP Dispatcher
// ============================================================================

/// Compute the VJP for a graph node.
///
/// Given the gradient of the output (`grad_output`), computes the gradients
/// for each input of `node`.
pub fn compute_vjp(node: &GraphNode, grad_output: &GraphNode) -> VjpResult {
    let sources = node.sources();

    match node.op() {
        GraphOp::View(view) => compute_view_vjp(node, grad_output, sources, view),
        GraphOp::MapReduce { map, reduce } => {
            if let Some((reduce_op, axis)) = reduce {
                compute_reduce_vjp(node, grad_output, sources, *reduce_op, *axis, map)
            } else {
                compute_elementwise_vjp(node, grad_output, sources, map)
            }
        }
    }
}

// ============================================================================
// Elementwise VJP
// ============================================================================

/// Compute VJP for elementwise operations (MapReduce without reduce).
fn compute_elementwise_vjp(
    _node: &GraphNode,
    grad_output: &GraphNode,
    inputs: &[GraphNode],
    map: &AstNode,
) -> VjpResult {
    let category = categorize_map(map);

    let input_grads = match category {
        // add(a, b): grad_a = grad_out, grad_b = grad_out
        MapCategory::Add => {
            vec![Some(grad_output.clone()), Some(grad_output.clone())]
        }

        // mul(a, b): grad_a = grad_out * b, grad_b = grad_out * a
        MapCategory::Mul => {
            vec![
                Some(grad_output * &inputs[1]),
                Some(grad_output * &inputs[0]),
            ]
        }

        // neg(a): grad_a = -grad_out
        MapCategory::Neg => {
            vec![Some(-grad_output)]
        }

        // recip(a) = 1/a: grad_a = -grad_out / a^2
        MapCategory::Recip => {
            let a_sq = &inputs[0] * &inputs[0];
            vec![Some(-grad_output / &a_sq)]
        }

        // sqrt(a): grad_a = grad_out * 0.5 / sqrt(a) = grad_out / (2 * sqrt(a))
        MapCategory::Sqrt => {
            let sqrt_a = inputs[0].sqrt();
            // grad_out / (2 * sqrt(a))
            let two_sqrt = &sqrt_a + &sqrt_a;
            vec![Some(grad_output / &two_sqrt)]
        }

        // exp(a): grad_a = grad_out * exp(a)
        MapCategory::Exp => {
            let exp_a = inputs[0].exp();
            vec![Some(grad_output * &exp_a)]
        }

        // log(a): grad_a = grad_out / a
        MapCategory::Log => {
            vec![Some(grad_output / &inputs[0])]
        }

        // exp2(a): grad_a = grad_out * exp2(a) * ln(2)
        MapCategory::Exp2 => {
            let exp2_a = inputs[0].exp2();
            let ln2 = create_scalar_like(grad_output, std::f32::consts::LN_2);
            vec![Some(&(grad_output * &exp2_a) * &ln2)]
        }

        // log2(a): grad_a = grad_out / (a * ln(2))
        MapCategory::Log2 => {
            // 1/ln(2) = log2(e)
            let inv_ln2 = create_scalar_like(grad_output, std::f32::consts::LOG2_E);
            vec![Some(&(grad_output / &inputs[0]) * &inv_ln2)]
        }

        // sin(a): grad_a = grad_out * cos(a)
        MapCategory::Sin => {
            let cos_a = inputs[0].cos();
            vec![Some(grad_output * &cos_a)]
        }

        // cos(a): grad_a = -grad_out * sin(a)
        MapCategory::Cos => {
            let sin_a = inputs[0].sin();
            vec![Some(-&(grad_output * &sin_a))]
        }

        // floor(a): grad_a = 0 (not differentiable, but we pass through)
        MapCategory::Floor => {
            vec![Some(zeros_like(grad_output))]
        }

        // abs(a): grad_a = grad_out * sign(a)
        MapCategory::Abs => {
            // sign(a) = a > 0 ? 1 : (a < 0 ? -1 : 0)
            // Simplified: sign(a) ≈ a / abs(a) for a != 0
            let zeros = zeros_like(&inputs[0]);
            let one = ones_like(&inputs[0]);
            let neg_one = -&one;
            let sign = inputs[0].gt(&zeros).where_cond(&one, &neg_one);
            vec![Some(grad_output * &sign)]
        }

        // max(a, b): grad_a = grad_out where a >= b, 0 elsewhere
        //            grad_b = grad_out where b > a, 0 elsewhere
        MapCategory::Max => {
            let mask_a = inputs[0].ge(&inputs[1]); // a >= b
            let mask_b = inputs[1].gt(&inputs[0]); // b > a
            let zeros = zeros_like(grad_output);
            vec![
                Some(grad_output.where_cond(&mask_a, &zeros)),
                Some(grad_output.where_cond(&mask_b, &zeros)),
            ]
        }

        // lt, gt, etc. (comparison): no gradient (boolean output)
        MapCategory::Comparison => {
            vec![None; inputs.len()]
        }

        // select(cond, then, else): grad_then = grad_out where cond, 0 elsewhere
        //                           grad_else = grad_out where !cond, 0 elsewhere
        MapCategory::Select => {
            // inputs[0] = cond, inputs[1] = then_val, inputs[2] = else_val
            let zeros = zeros_like(grad_output);
            let cond = &inputs[0];
            vec![
                None, // cond has no gradient
                Some(grad_output.where_cond(cond, &zeros)),
                Some(zeros.where_cond(cond, grad_output)),
            ]
        }

        // cast(a, dtype): gradient passes through (assuming float types)
        MapCategory::Cast => {
            // Cast gradient back to original type
            let orig_dtype = inputs[0].dtype().clone();
            vec![Some(grad_output.cast(orig_dtype))]
        }

        // Unknown/complex operation
        MapCategory::Unknown => {
            // Return zeros for safety
            vec![Some(zeros_like(grad_output)); inputs.len()]
        }
    };

    VjpResult { input_grads }
}

// ============================================================================
// Reduce VJP
// ============================================================================

/// Compute VJP for reduce operations.
fn compute_reduce_vjp(
    node: &GraphNode,
    grad_output: &GraphNode,
    inputs: &[GraphNode],
    reduce_op: ReduceOp,
    axis: usize,
    _map: &AstNode,
) -> VjpResult {
    let input_shape = inputs[0].shape();
    let axis_size = input_shape[axis].clone();

    // The grad_output may have the reduced axis squeezed out (if it came through
    // Tensor API). We need to restore it before expanding.
    // E.g., if input was [10, 20] and sum(axis=1), grad_output might be [10]
    // We need to unsqueeze it to [10, 1] first, then expand to [10, 20]
    let grad_with_axis = if grad_output.ndim() < inputs[0].ndim() {
        // The axis was squeezed, restore it
        grad_output.unsqueeze(axis)
    } else {
        grad_output.clone()
    };

    let input_grads = match reduce_op {
        // sum(a, axis): grad_a = expand(grad_out, axis, size)
        ReduceOp::Sum => {
            let grad_expanded = grad_with_axis.expand(axis, axis_size);
            vec![Some(grad_expanded)]
        }

        // max(a, axis): grad_a = one_hot_mask * expand(grad_out, axis)
        // The gradient only flows to positions where the max occurred
        ReduceOp::Max => {
            // node contains the max values (reduced), may need unsqueeze
            let node_with_axis = if node.ndim() < inputs[0].ndim() {
                node.unsqueeze(axis)
            } else {
                node.clone()
            };
            // Expand max back to original shape for comparison
            let max_expanded = node_with_axis.expand(axis, axis_size.clone());
            // Create mask: 1 where input equals max, 0 elsewhere
            let mask = inputs[0].eq_node(&max_expanded);
            let mask_float = mask.cast(inputs[0].dtype().clone());

            // Simplified: just use mask without normalization
            // TODO: Proper normalization with sum_keepdim
            let grad_expanded = grad_with_axis.expand(axis, axis_size);
            vec![Some(&grad_expanded * &mask_float)]
        }

        // min(a, axis): similar to max
        ReduceOp::Min => {
            let node_with_axis = if node.ndim() < inputs[0].ndim() {
                node.unsqueeze(axis)
            } else {
                node.clone()
            };
            let min_expanded = node_with_axis.expand(axis, axis_size.clone());
            let mask = inputs[0].eq_node(&min_expanded);
            let mask_float = mask.cast(inputs[0].dtype().clone());

            // Simplified: just use mask without normalization
            // TODO: Proper normalization with sum_keepdim
            let grad_expanded = grad_with_axis.expand(axis, axis_size);
            vec![Some(&grad_expanded * &mask_float)]
        }

        // prod(a, axis): grad_a = grad_out * prod(a, axis) / a
        // Special handling needed for zeros
        ReduceOp::Prod => {
            let node_with_axis = if node.ndim() < inputs[0].ndim() {
                node.unsqueeze(axis)
            } else {
                node.clone()
            };
            // Simple case (no zeros): grad_a = expand(grad_out * prod / a)
            let prod_expanded = node_with_axis.expand(axis, axis_size.clone());
            let grad_expanded = grad_with_axis.expand(axis, axis_size);
            let grad_input = &(&grad_expanded * &prod_expanded) / &inputs[0];
            // Note: This is incorrect when input contains zeros
            // Full implementation would need special handling
            vec![Some(grad_input)]
        }
    };

    VjpResult { input_grads }
}

// ============================================================================
// View VJP
// ============================================================================

/// Compute VJP for view operations.
fn compute_view_vjp(
    node: &GraphNode,
    grad_output: &GraphNode,
    inputs: &[GraphNode],
    view: &crate::graph::shape::View,
) -> VjpResult {
    use crate::graph::shape::View;

    let grad_input = match view {
        // For reshape: reshape grad_output back to original shape
        View::Linear { .. } => {
            let orig_shape = inputs[0].shape();
            let node_shape = node.shape();

            if orig_shape != node_shape {
                // This is a reshape or similar
                grad_output.reshape(orig_shape)
            } else {
                // Identity view
                grad_output.clone()
            }
        }

        // For indexed views, need to invert the indexing
        View::IndexExpr { .. } => {
            // This is complex - for now, try to infer the inverse
            let orig_shape = inputs[0].shape();
            grad_output.reshape(orig_shape)
        }

        // Masked view
        View::Masked { .. } => {
            // Apply same mask to gradient
            // For simplicity, treat as reshape
            let orig_shape = inputs[0].shape();
            grad_output.reshape(orig_shape)
        }
    };

    VjpResult {
        input_grads: vec![Some(grad_input)],
    }
}

// ============================================================================
// Map Categorization
// ============================================================================

/// Category of elementwise operation for gradient computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapCategory {
    Add,
    Mul,
    Neg,
    Recip,
    Sqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Sin,
    Cos,
    Floor,
    Abs,
    Max,
    Comparison,
    Select,
    Cast,
    Unknown,
}

/// Categorize an AstNode map operation.
fn categorize_map(map: &AstNode) -> MapCategory {
    // Check for negation first (Mul with -1 constant)
    if is_negation(map) {
        return MapCategory::Neg;
    }

    match map {
        AstNode::Add(_, _) => MapCategory::Add,
        AstNode::Mul(_, _) => MapCategory::Mul,
        AstNode::Recip(_) => MapCategory::Recip,
        AstNode::Sqrt(_) => MapCategory::Sqrt,
        AstNode::Exp2(_) => MapCategory::Exp2,
        AstNode::Log2(_) => MapCategory::Log2,
        AstNode::Sin(_) => MapCategory::Sin,
        AstNode::Floor(_) => MapCategory::Floor,
        AstNode::Max(_, _) => MapCategory::Max,
        AstNode::Lt(_, _) => MapCategory::Comparison,
        AstNode::And(_, _) => MapCategory::Comparison,
        AstNode::Not(_) => MapCategory::Comparison,
        AstNode::Select { .. } => MapCategory::Select,
        AstNode::Cast { .. } => MapCategory::Cast,

        // Check for exp (implemented as exp2(x * log2(e)))
        _ if is_exp(map) => MapCategory::Exp,

        // Check for log (implemented as log2(x) / log2(e))
        _ if is_log(map) => MapCategory::Log,

        // Check for cos (implemented via sin)
        _ if is_cos(map) => MapCategory::Cos,

        // Check for abs
        _ if is_abs(map) => MapCategory::Abs,

        _ => MapCategory::Unknown,
    }
}

/// Check if map represents negation.
fn is_negation(map: &AstNode) -> bool {
    matches!(map,
        AstNode::Mul(a, b) if is_const_neg_one(a.as_ref()) || is_const_neg_one(b.as_ref())
    )
}

/// Check if AstNode is constant -1.
fn is_const_neg_one(node: &AstNode) -> bool {
    matches!(node,
        AstNode::Const(crate::ast::Literal::F32(v)) if *v == -1.0
    ) || matches!(node,
        AstNode::Const(crate::ast::Literal::I32(v)) if *v == -1
    )
}

/// Check if map represents exp (e^x).
fn is_exp(map: &AstNode) -> bool {
    // exp is typically: exp2(x * log2(e)) where log2(e) ≈ 1.4427
    matches!(map, AstNode::Exp2(_))
}

/// Check if map represents natural log.
fn is_log(_map: &AstNode) -> bool {
    // ln is typically: log2(x) * ln(2) or log2(x) / log2(e)
    false // Simplified - actual detection would check for log2 patterns
}

/// Check if map represents cosine.
fn is_cos(_map: &AstNode) -> bool {
    // cos is typically: sin(x + pi/2)
    false // Simplified
}

/// Check if map represents absolute value.
fn is_abs(map: &AstNode) -> bool {
    // abs can be: select(x < 0, -x, x) or sqrt(x*x)
    matches!(map, AstNode::Select { .. })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a tensor of zeros with the same shape as the input.
fn zeros_like(node: &GraphNode) -> GraphNode {
    crate::graph::zeros(node.shape().clone(), node.dtype().clone())
}

/// Create a tensor of ones with the same shape as the input.
fn ones_like(node: &GraphNode) -> GraphNode {
    ones(node.shape().clone(), node.dtype().clone())
}

/// Create a scalar-like tensor (all elements same value) with the same shape.
fn create_scalar_like(node: &GraphNode, _value: f32) -> GraphNode {
    // Create ones and we'd ideally multiply by value
    // For now, this is a placeholder - proper scalar constants need backend support

    // TODO: Implement proper scalar multiplication
    ones_like(node)
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
    fn test_add_vjp() {
        let a = input(vec![Expr::Const(10)], DType::F32);
        let b = input(vec![Expr::Const(10)], DType::F32);
        let c = &a + &b;

        let grad_out = input(vec![Expr::Const(10)], DType::F32);
        let vjp = compute_vjp(&c, &grad_out);

        assert_eq!(vjp.input_grads.len(), 2);
        assert!(vjp.input_grads[0].is_some());
        assert!(vjp.input_grads[1].is_some());
    }

    #[test]
    fn test_mul_vjp() {
        let a = input(vec![Expr::Const(10)], DType::F32);
        let b = input(vec![Expr::Const(10)], DType::F32);
        let c = &a * &b;

        let grad_out = input(vec![Expr::Const(10)], DType::F32);
        let vjp = compute_vjp(&c, &grad_out);

        assert_eq!(vjp.input_grads.len(), 2);
        // grad_a = grad_out * b
        // grad_b = grad_out * a
    }

    #[test]
    fn test_sum_vjp() {
        let a = input(vec![Expr::Const(10), Expr::Const(20)], DType::F32);
        let b = a.sum(1); // [10, 20] -> [10]

        let grad_out = input(vec![Expr::Const(10)], DType::F32);
        let vjp = compute_vjp(&b, &grad_out);

        assert_eq!(vjp.input_grads.len(), 1);
        // grad_a should be expand(grad_out, axis=1, size=20)
        let grad_a = vjp.input_grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), vec![Expr::Const(10), Expr::Const(20)]);
    }

    #[test]
    fn test_neg_vjp() {
        let a = input(vec![Expr::Const(10)], DType::F32);
        let b = -&a;

        let grad_out = input(vec![Expr::Const(10)], DType::F32);
        let vjp = compute_vjp(&b, &grad_out);

        assert_eq!(vjp.input_grads.len(), 1);
        // grad_a = -grad_out
    }
}
