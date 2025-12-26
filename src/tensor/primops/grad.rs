//! Gradient functions for primitive operations
//!
//! This module provides:
//! - Helper functions for gradient computation
//! - CloneBackward (general utility)
//! - Symbolic differentiation for fused operations
//! - FusedElementwiseBackward and FusedElementwiseReduceBackward
//!
//! Individual operation gradients are defined in their respective modules:
//! - binary.rs: AddBackward, MulBackward, MaxBackward
//! - unary.rs: NegBackward, RecipBackward, SqrtBackward, Log2Backward, Exp2Backward, SinBackward
//! - reduce.rs: ReduceSumBackward, ReduceMulBackward, ReduceMaxBackward
//!
//! Note: The GradFn trait and Backward structs are currently f32-only because:
//! - GradFn is used as a trait object (dyn GradFn) which doesn't support type parameters
//! - AutogradMeta stores grad_fn as Arc<dyn GradFn>
//! Full FloatDType support would require architectural changes to the autograd system.

use crate::ast::{AstNode, Literal};
use crate::tensor::ops::ReduceOp;
use crate::tensor::{DimDyn, Exp2, Floor, GradFn, Log2, Recip, Sin, Sqrt, Tensor};
use std::collections::HashMap;

// ============================================================================
// Helper Functions
// ============================================================================

/// Reduce gradient to match the original input shape (handle broadcasting)
///
/// Note: Currently f32-only because:
/// - GradFn trait is f32-only
/// - reduce_sum and reshape_dyn would need FloatDType implementations first
pub fn reduce_grad_for_broadcast(
    grad: &Tensor<f32, DimDyn>,
    target_shape: &[usize],
) -> Tensor<f32, DimDyn> {
    if grad.shape() == target_shape {
        return grad.clone();
    }

    let grad_shape = grad.shape();
    let target_ndim = target_shape.len();
    let grad_ndim = grad_shape.len();

    // Pad target shape with 1s on the left to match grad_ndim
    let mut padded_target = vec![1usize; grad_ndim.saturating_sub(target_ndim)];
    padded_target.extend_from_slice(target_shape);

    // Find axes to reduce
    let mut reduce_axes = Vec::new();
    for (i, (&grad_dim, &target_dim)) in grad_shape.iter().zip(padded_target.iter()).enumerate() {
        if target_dim == 1 && grad_dim > 1 {
            reduce_axes.push(i);
        }
    }

    // Reduce along those axes
    let mut result = grad.clone();
    if !reduce_axes.is_empty() {
        result = result.reduce_sum(&reduce_axes, true);
    }

    // Reshape to target shape
    result.reshape_dyn(target_shape)
}

// ============================================================================
// Clone Gradient (general utility)
// ============================================================================

/// Gradient for Clone (fork): z = clone(a)
/// ∂L/∂a = ∂L/∂z (identity - gradient passes through)
pub struct CloneBackward {
    input: Tensor<f32, DimDyn>,
}

impl CloneBackward {
    pub fn new(input: Tensor<f32, DimDyn>) -> Self {
        Self { input }
    }
}

impl GradFn for CloneBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // Clone is identity for gradients - just pass through
        vec![grad_output.clone()]
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "CloneBackward"
    }
}

// ============================================================================
// Symbolic Differentiation for Fused Operations
// ============================================================================

/// Compute symbolic derivative of an AstNode expression with respect to a Wildcard
///
/// This implements automatic differentiation at the AST level for fused operations.
/// The result is another AstNode representing ∂expr/∂Wildcard(wrt).
fn differentiate_ast(expr: &AstNode, wrt: &str) -> AstNode {
    match expr {
        // Constant: ∂c/∂x = 0
        AstNode::Const(_) => AstNode::Const(Literal::F32(0.0)),

        // Wildcard: ∂x/∂x = 1, ∂y/∂x = 0
        AstNode::Wildcard(name) => {
            if name == wrt {
                AstNode::Const(Literal::F32(1.0))
            } else {
                AstNode::Const(Literal::F32(0.0))
            }
        }

        // Addition: ∂(a + b)/∂x = ∂a/∂x + ∂b/∂x
        AstNode::Add(lhs, rhs) => {
            let d_lhs = differentiate_ast(lhs, wrt);
            let d_rhs = differentiate_ast(rhs, wrt);
            AstNode::Add(Box::new(d_lhs), Box::new(d_rhs))
        }

        // Multiplication: ∂(a * b)/∂x = a * ∂b/∂x + b * ∂a/∂x
        AstNode::Mul(lhs, rhs) => {
            let d_lhs = differentiate_ast(lhs, wrt);
            let d_rhs = differentiate_ast(rhs, wrt);
            // a * d_rhs + b * d_lhs
            AstNode::Add(
                Box::new(AstNode::Mul(Box::new((**lhs).clone()), Box::new(d_rhs))),
                Box::new(AstNode::Mul(Box::new((**rhs).clone()), Box::new(d_lhs))),
            )
        }

        // Reciprocal: ∂(1/a)/∂x = -∂a/∂x / a²
        AstNode::Recip(inner) => {
            let d_inner = differentiate_ast(inner, wrt);
            // -d_inner * recip(inner)^2
            let recip_squared = AstNode::Mul(
                Box::new(AstNode::Recip(inner.clone())),
                Box::new(AstNode::Recip(inner.clone())),
            );
            AstNode::Mul(
                Box::new(AstNode::Mul(
                    Box::new(AstNode::Const(Literal::F32(-1.0))),
                    Box::new(d_inner),
                )),
                Box::new(recip_squared),
            )
        }

        // Sqrt: ∂√a/∂x = ∂a/∂x / (2 * √a)
        AstNode::Sqrt(inner) => {
            let d_inner = differentiate_ast(inner, wrt);
            // d_inner / (2 * sqrt(inner))
            let two_sqrt = AstNode::Mul(
                Box::new(AstNode::Const(Literal::F32(2.0))),
                Box::new(AstNode::Sqrt(inner.clone())),
            );
            AstNode::Mul(
                Box::new(d_inner),
                Box::new(AstNode::Recip(Box::new(two_sqrt))),
            )
        }

        // Log2: ∂log₂(a)/∂x = ∂a/∂x / (a * ln(2))
        AstNode::Log2(inner) => {
            let d_inner = differentiate_ast(inner, wrt);
            let ln2 = std::f32::consts::LN_2;
            // d_inner / (inner * ln2)
            let denominator = AstNode::Mul(
                Box::new((**inner).clone()),
                Box::new(AstNode::Const(Literal::F32(ln2))),
            );
            AstNode::Mul(
                Box::new(d_inner),
                Box::new(AstNode::Recip(Box::new(denominator))),
            )
        }

        // Exp2: ∂2^a/∂x = ∂a/∂x * 2^a * ln(2)
        AstNode::Exp2(inner) => {
            let d_inner = differentiate_ast(inner, wrt);
            let ln2 = std::f32::consts::LN_2;
            // d_inner * exp2(inner) * ln2
            AstNode::Mul(
                Box::new(AstNode::Mul(
                    Box::new(d_inner),
                    Box::new(AstNode::Exp2(inner.clone())),
                )),
                Box::new(AstNode::Const(Literal::F32(ln2))),
            )
        }

        // Sin: ∂sin(a)/∂x = ∂a/∂x * cos(a) = ∂a/∂x * sin(a + π/2)
        AstNode::Sin(inner) => {
            let d_inner = differentiate_ast(inner, wrt);
            let pi_2 = std::f32::consts::FRAC_PI_2;
            // d_inner * sin(inner + pi/2)
            let cos_inner = AstNode::Sin(Box::new(AstNode::Add(
                inner.clone(),
                Box::new(AstNode::Const(Literal::F32(pi_2))),
            )));
            AstNode::Mul(Box::new(d_inner), Box::new(cos_inner))
        }

        // Floor: floor is non-differentiable (gradient is 0 almost everywhere)
        AstNode::Floor(_) => AstNode::Const(Literal::F32(0.0)),

        // Max: subdifferential - we approximate by assuming left operand dominates
        // This is a simplification; proper implementation needs comparison ops
        AstNode::Max(lhs, _rhs) => {
            // Approximate: gradient flows to lhs only
            differentiate_ast(lhs, wrt)
        }

        // FMA: ∂(a * b + c)/∂x = b * ∂a/∂x + a * ∂b/∂x + ∂c/∂x
        AstNode::Fma { a, b, c } => {
            let d_a = differentiate_ast(a, wrt);
            let d_b = differentiate_ast(b, wrt);
            let d_c = differentiate_ast(c, wrt);
            // b * d_a + a * d_b + d_c
            AstNode::Add(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Mul(Box::new((**b).clone()), Box::new(d_a))),
                    Box::new(AstNode::Mul(Box::new((**a).clone()), Box::new(d_b))),
                )),
                Box::new(d_c),
            )
        }

        // Cast: pass through derivative (assuming compatible types)
        AstNode::Cast(inner, dtype) => {
            AstNode::Cast(Box::new(differentiate_ast(inner, wrt)), dtype.clone())
        }

        // Rem, Idiv: non-differentiable, return 0
        AstNode::Rem(_, _) | AstNode::Idiv(_, _) => AstNode::Const(Literal::F32(0.0)),

        // Other nodes: not supported in elementwise expressions, return 0
        _ => AstNode::Const(Literal::F32(0.0)),
    }
}

/// Find all wildcard names in an AstNode expression
fn find_wildcards(expr: &AstNode) -> Vec<String> {
    let mut wildcards = Vec::new();
    find_wildcards_impl(expr, &mut wildcards);
    wildcards.sort();
    wildcards.dedup();
    wildcards
}

fn find_wildcards_impl(expr: &AstNode, result: &mut Vec<String>) {
    match expr {
        AstNode::Wildcard(name) => {
            result.push(name.clone());
        }
        _ => {
            for child in expr.children() {
                find_wildcards_impl(child, result);
            }
        }
    }
}

// ============================================================================
// Fused Operation Gradients
// ============================================================================

/// Gradient for FusedElementwise operations
///
/// Computes gradients for each input by symbolically differentiating the expression
/// and then evaluating the derivative with the actual input values.
pub struct FusedElementwiseBackward {
    inputs: Vec<Tensor<f32, DimDyn>>,
    /// Precomputed derivative expressions for each input wildcard
    grad_exprs: Vec<(String, AstNode)>,
}

impl FusedElementwiseBackward {
    pub fn new(inputs: Vec<Tensor<f32, DimDyn>>, expr: AstNode) -> Self {
        // Find all wildcards and compute derivatives
        let wildcards = find_wildcards(&expr);
        let grad_exprs: Vec<(String, AstNode)> = wildcards
            .iter()
            .map(|wc| (wc.clone(), differentiate_ast(&expr, wc)))
            .collect();

        Self { inputs, grad_exprs }
    }
}

impl GradFn for FusedElementwiseBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // Build substitution map from wildcard names to input tensors
        let mut substitution_map: HashMap<String, Tensor<f32, DimDyn>> = HashMap::new();
        for (i, input) in self.inputs.iter().enumerate() {
            substitution_map.insert(i.to_string(), input.clone());
        }

        // Compute gradient for each input
        let mut grads = Vec::new();
        for (wc_name, grad_expr) in &self.grad_exprs {
            // Evaluate the gradient expression with the actual input values
            let grad = evaluate_ast_with_tensors(grad_expr, &substitution_map);

            // Multiply by incoming gradient
            let full_grad = grad_output * &grad;

            // Reduce for broadcasting if needed
            if let Ok(idx) = wc_name.parse::<usize>()
                && idx < self.inputs.len()
            {
                let input_grad = reduce_grad_for_broadcast(&full_grad, self.inputs[idx].shape());
                grads.push(input_grad);
            }
        }

        // Ensure we return gradients for all inputs
        while grads.len() < self.inputs.len() {
            // This shouldn't happen normally, but provide zero gradients as fallback
            let idx = grads.len();
            grads.push(Tensor::<f32, DimDyn>::zeros_dyn(self.inputs[idx].shape()));
        }

        grads
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        self.inputs.clone()
    }

    fn name(&self) -> &'static str {
        "FusedElementwiseBackward"
    }
}

/// Gradient for FusedElementwiseReduce operations
///
/// Handles the case where elementwise operations are fused with a reduce operation.
pub struct FusedElementwiseReduceBackward {
    inputs: Vec<Tensor<f32, DimDyn>>,
    input_shapes: Vec<Vec<usize>>,
    #[allow(dead_code)]
    reduce_op: ReduceOp, // TODO: Use for proper gradient computation (e.g., ReduceMax mask)
    axes: Vec<usize>,
    keepdim: bool,
    /// Precomputed derivative expressions for each input wildcard
    grad_exprs: Vec<(String, AstNode)>,
}

impl FusedElementwiseReduceBackward {
    pub fn new(
        inputs: Vec<Tensor<f32, DimDyn>>,
        expr: AstNode,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Self {
        let input_shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape().to_vec()).collect();

        // Find all wildcards and compute derivatives
        let wildcards = find_wildcards(&expr);
        let grad_exprs: Vec<(String, AstNode)> = wildcards
            .iter()
            .map(|wc| (wc.clone(), differentiate_ast(&expr, wc)))
            .collect();

        Self {
            inputs,
            input_shapes,
            reduce_op,
            axes,
            keepdim,
            grad_exprs,
        }
    }
}

impl GradFn for FusedElementwiseReduceBackward {
    fn backward(&self, grad_output: &Tensor<f32, DimDyn>) -> Vec<Tensor<f32, DimDyn>> {
        // First, expand grad_output to match the pre-reduction shape
        let mut grad = grad_output.clone();
        if !self.keepdim {
            for &axis in &self.axes {
                grad = grad.unsqueeze(axis);
            }
        }

        // Find the largest input shape to expand to
        let max_shape = self
            .input_shapes
            .iter()
            .max_by_key(|s| s.len())
            .cloned()
            .unwrap_or_default();
        let grad_expanded = grad.expand(&max_shape);

        // Build substitution map
        let mut substitution_map: HashMap<String, Tensor<f32, DimDyn>> = HashMap::new();
        for (i, input) in self.inputs.iter().enumerate() {
            substitution_map.insert(i.to_string(), input.clone());
        }

        // Compute gradient for each input
        let mut grads = Vec::new();
        for (wc_name, grad_expr) in &self.grad_exprs {
            // Evaluate the gradient expression
            let local_grad = evaluate_ast_with_tensors(grad_expr, &substitution_map);

            // Multiply by expanded incoming gradient
            let full_grad = &grad_expanded * &local_grad;

            // Reduce for broadcasting if needed
            if let Ok(idx) = wc_name.parse::<usize>()
                && idx < self.inputs.len()
            {
                let input_grad = reduce_grad_for_broadcast(&full_grad, self.inputs[idx].shape());
                grads.push(input_grad);
            }
        }

        // Ensure we return gradients for all inputs
        while grads.len() < self.inputs.len() {
            let idx = grads.len();
            grads.push(Tensor::<f32, DimDyn>::zeros_dyn(self.inputs[idx].shape()));
        }

        grads
    }

    fn inputs(&self) -> Vec<Tensor<f32, DimDyn>> {
        self.inputs.clone()
    }

    fn name(&self) -> &'static str {
        "FusedElementwiseReduceBackward"
    }
}

/// Evaluate an AstNode expression with tensor values substituted for wildcards
fn evaluate_ast_with_tensors(
    expr: &AstNode,
    substitution: &HashMap<String, Tensor<f32, DimDyn>>,
) -> Tensor<f32, DimDyn> {
    match expr {
        AstNode::Const(Literal::F32(val)) => {
            // Create a scalar tensor - shape will be broadcast during operations
            Tensor::<f32, DimDyn>::full_dyn(&[1], *val)
        }
        AstNode::Const(Literal::I32(val)) => Tensor::<f32, DimDyn>::full_dyn(&[1], *val as f32),

        AstNode::Wildcard(name) => substitution
            .get(name)
            .cloned()
            .unwrap_or_else(|| Tensor::<f32, DimDyn>::zeros_dyn(&[1])),

        AstNode::Add(lhs, rhs) => {
            let l = evaluate_ast_with_tensors(lhs, substitution);
            let r = evaluate_ast_with_tensors(rhs, substitution);
            &l + &r
        }

        AstNode::Mul(lhs, rhs) => {
            let l = evaluate_ast_with_tensors(lhs, substitution);
            let r = evaluate_ast_with_tensors(rhs, substitution);
            &l * &r
        }

        AstNode::Recip(inner) => {
            let i = evaluate_ast_with_tensors(inner, substitution);
            i.recip()
        }

        AstNode::Sqrt(inner) => {
            let i = evaluate_ast_with_tensors(inner, substitution);
            i.sqrt()
        }

        AstNode::Log2(inner) => {
            let i = evaluate_ast_with_tensors(inner, substitution);
            i.log2()
        }

        AstNode::Exp2(inner) => {
            let i = evaluate_ast_with_tensors(inner, substitution);
            i.exp2()
        }

        AstNode::Sin(inner) => {
            let i = evaluate_ast_with_tensors(inner, substitution);
            i.sin()
        }

        AstNode::Floor(inner) => {
            let i = evaluate_ast_with_tensors(inner, substitution);
            i.floor()
        }

        AstNode::Max(lhs, rhs) => {
            let l = evaluate_ast_with_tensors(lhs, substitution);
            let r = evaluate_ast_with_tensors(rhs, substitution);
            l.max(&r)
        }

        AstNode::Cast(inner, _dtype) => {
            // For now, just evaluate inner (cast is identity for f32)
            evaluate_ast_with_tensors(inner, substitution)
        }

        AstNode::Fma { a, b, c } => {
            let av = evaluate_ast_with_tensors(a, substitution);
            let bv = evaluate_ast_with_tensors(b, substitution);
            let cv = evaluate_ast_with_tensors(c, substitution);
            &(&av * &bv) + &cv
        }

        // For unsupported nodes, return zeros
        _ => Tensor::<f32, DimDyn>::zeros_dyn(&[1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::wildcard;
    use crate::tensor::primops::binary::AddBackward;

    #[test]
    fn test_add_backward_name() {
        let t = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let backward = AddBackward::new(t.clone(), t.clone());
        assert_eq!(backward.name(), "AddBackward");
    }

    #[test]
    fn test_reduce_grad_for_broadcast_same_shape() {
        let grad = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let result = reduce_grad_for_broadcast(&grad, &[2, 3]);
        assert_eq!(result.shape(), &[2, 3]);
    }

    // ============================================================================
    // Symbolic Differentiation Tests
    // ============================================================================

    #[test]
    fn test_differentiate_constant() {
        let expr = AstNode::Const(Literal::F32(5.0));
        let grad = differentiate_ast(&expr, "0");
        match grad {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 0.0),
            _ => panic!("Expected Const(0.0)"),
        }
    }

    #[test]
    fn test_differentiate_wildcard_self() {
        let expr = wildcard("0");
        let grad = differentiate_ast(&expr, "0");
        match grad {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 1.0),
            _ => panic!("Expected Const(1.0)"),
        }
    }

    #[test]
    fn test_differentiate_wildcard_other() {
        let expr = wildcard("0");
        let grad = differentiate_ast(&expr, "1");
        match grad {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 0.0),
            _ => panic!("Expected Const(0.0)"),
        }
    }

    #[test]
    fn test_differentiate_addition() {
        // d/dx(x + y) = 1 + 0 = 1
        let expr = AstNode::Add(Box::new(wildcard("0")), Box::new(wildcard("1")));
        let grad = differentiate_ast(&expr, "0");
        // Should be Add(Const(1.0), Const(0.0))
        match grad {
            AstNode::Add(lhs, rhs) => match (*lhs, *rhs) {
                (AstNode::Const(Literal::F32(l)), AstNode::Const(Literal::F32(r))) => {
                    assert_eq!(l, 1.0);
                    assert_eq!(r, 0.0);
                }
                _ => panic!("Expected Const values"),
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_differentiate_multiplication() {
        // d/dx(x * y) = y * 1 + x * 0 = y
        let expr = AstNode::Mul(Box::new(wildcard("0")), Box::new(wildcard("1")));
        let grad = differentiate_ast(&expr, "0");
        // Result is Add(Mul(x, Const(0)), Mul(y, Const(1)))
        // which simplifies to y
        match grad {
            AstNode::Add(_, _) => {} // Expected structure
            _ => panic!("Expected Add node for product rule"),
        }
    }

    #[test]
    fn test_find_wildcards() {
        // Expression: x + y * z
        let expr = AstNode::Add(
            Box::new(wildcard("0")),
            Box::new(AstNode::Mul(
                Box::new(wildcard("1")),
                Box::new(wildcard("2")),
            )),
        );
        let wildcards = find_wildcards(&expr);
        assert_eq!(wildcards, vec!["0", "1", "2"]);
    }

    #[test]
    fn test_find_wildcards_with_duplicates() {
        // Expression: x + x * y
        let expr = AstNode::Add(
            Box::new(wildcard("0")),
            Box::new(AstNode::Mul(
                Box::new(wildcard("0")),
                Box::new(wildcard("1")),
            )),
        );
        let wildcards = find_wildcards(&expr);
        assert_eq!(wildcards, vec!["0", "1"]); // Duplicates removed
    }

    // ============================================================================
    // Fused Backward Tests
    // ============================================================================

    #[test]
    fn test_fused_elementwise_backward_new() {
        let t1 = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let t2 = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let expr = AstNode::Add(Box::new(wildcard("0")), Box::new(wildcard("1")));

        let backward = FusedElementwiseBackward::new(vec![t1, t2], expr);
        assert_eq!(backward.name(), "FusedElementwiseBackward");
        assert_eq!(backward.inputs().len(), 2);
        assert_eq!(backward.grad_exprs.len(), 2);
    }

    #[test]
    fn test_fused_elementwise_reduce_backward_new() {
        let t1 = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let t2 = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let expr = AstNode::Mul(Box::new(wildcard("0")), Box::new(wildcard("1")));

        let backward =
            FusedElementwiseReduceBackward::new(vec![t1, t2], expr, ReduceOp::Sum, vec![1], false);
        assert_eq!(backward.name(), "FusedElementwiseReduceBackward");
        assert_eq!(backward.inputs().len(), 2);
        assert_eq!(backward.axes, vec![1]);
    }

    #[test]
    fn test_evaluate_ast_constant() {
        let expr = AstNode::Const(Literal::F32(3.5));
        let substitution = HashMap::new();
        let result = evaluate_ast_with_tensors(&expr, &substitution);
        assert_eq!(result.shape(), &[1]);
    }

    #[test]
    fn test_evaluate_ast_wildcard() {
        let expr = wildcard("0");
        let mut substitution = HashMap::new();
        let t = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        substitution.insert("0".to_string(), t);
        let result = evaluate_ast_with_tensors(&expr, &substitution);
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_evaluate_ast_addition() {
        let expr = AstNode::Add(Box::new(wildcard("0")), Box::new(wildcard("1")));
        let mut substitution = HashMap::new();
        let t1 = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        let t2 = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);
        substitution.insert("0".to_string(), t1);
        substitution.insert("1".to_string(), t2);
        let result = evaluate_ast_with_tensors(&expr, &substitution);
        assert_eq!(result.shape(), &[2, 3]);
    }
}
