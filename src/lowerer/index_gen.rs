//! Index expression generation from Views
//!
//! This module converts View structures into index expressions
//! that can be used in AST Load/Store operations.

use crate::ast::AstNode;
use crate::graph::shape::{Expr, PadValue, View};

/// Index generator for converting Views to AST expressions
pub struct IndexGenerator {
    /// Variable name prefix for loop indices
    idx_prefix: String,
}

impl Default for IndexGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexGenerator {
    /// Create a new index generator
    pub fn new() -> Self {
        Self {
            idx_prefix: "ridx".to_string(),
        }
    }

    /// Create index generator with custom prefix
    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            idx_prefix: prefix.into(),
        }
    }

    /// Generate loop index variable name for a dimension
    pub fn loop_var(&self, dim: usize) -> String {
        format!("{}{}", self.idx_prefix, dim)
    }

    /// Convert a View to an index expression (as Expr)
    ///
    /// The result is an Expr that computes the linear index into the
    /// underlying buffer given the loop variables ridx0, ridx1, ...
    pub fn view_to_expr(&self, view: &View) -> Expr {
        match view {
            View::Linear {
                strides, offset, ..
            } => {
                // Linear index: offset + sum(idx[i] * stride[i])
                let mut result = offset.clone();
                for (i, stride) in strides.iter().enumerate() {
                    let idx = Expr::Idx(i);
                    let term = idx * stride.clone();
                    result += term;
                }
                result
            }
            View::IndexExpr { index_expr, .. } => {
                // Use the index expression directly
                index_expr.clone()
            }
        }
    }

    /// Convert a View to an AstNode index expression
    pub fn view_to_index(&self, view: &View) -> AstNode {
        let expr = self.view_to_expr(view);
        expr.into() // Uses From<Expr> for AstNode
    }

    /// Check if view has bounds and return the bounds condition
    ///
    /// Returns the condition expression and default value if the view has bounds.
    pub fn get_bounds(&self, view: &View) -> Option<(Expr, PadValue)> {
        let bounds = view.bounds();
        if bounds.has_condition() {
            Some((bounds.condition.clone(), bounds.default_value))
        } else {
            None
        }
    }

    /// Convert an Expr to AstNode (convenience wrapper)
    pub fn expr_to_ast(&self, expr: &Expr) -> AstNode {
        expr.clone().into()
    }

    /// Generate loop bounds from shape
    pub fn shape_to_bounds(&self, shape: &[Expr]) -> Vec<(String, AstNode)> {
        shape
            .iter()
            .enumerate()
            .map(|(i, dim)| {
                let var = self.loop_var(i);
                let bound: AstNode = dim.clone().into();
                (var, bound)
            })
            .collect()
    }

    /// Convert a View to an index expression for reduce output
    ///
    /// For reduce operations, the reduced dimension's index should be 0
    /// (since the output size is 1 in that dimension).
    /// This substitutes Idx(reduce_axis) with Const(0).
    pub fn view_to_reduce_output_index(&self, view: &View, reduce_axis: usize) -> AstNode {
        let expr = self.view_to_expr(view);
        // Substitute the reduce axis index with 0
        let fixed_expr = expr.substitute_idx(reduce_axis, Expr::Const(0)).simplify();
        fixed_expr.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_index_simple() {
        let idx_gen = IndexGenerator::new();
        let view = View::contiguous(vec![Expr::Const(32), Expr::Const(64)]);

        let idx_expr = idx_gen.view_to_expr(&view);
        // Should produce: offset + ridx0 * 64 + ridx1 * 1
        // The exact structure depends on Expr simplification
        assert!(matches!(idx_expr, Expr::Add(_, _)));
    }

    #[test]
    fn test_shape_to_bounds() {
        let idx_gen = IndexGenerator::new();
        let shape = vec![Expr::Const(32), Expr::Const(64)];

        let bounds = idx_gen.shape_to_bounds(&shape);
        assert_eq!(bounds.len(), 2);
        assert_eq!(bounds[0].0, "ridx0");
        assert_eq!(bounds[1].0, "ridx1");
    }

    #[test]
    fn test_expr_to_ast() {
        let idx_gen = IndexGenerator::new();

        // Test constant
        let ast = idx_gen.expr_to_ast(&Expr::Const(42));
        assert!(matches!(ast, AstNode::Const(crate::ast::Literal::I64(42))));

        // Test index
        let ast = idx_gen.expr_to_ast(&Expr::Idx(0));
        assert!(matches!(ast, AstNode::Var(ref s) if s == "ridx0"));
    }

    #[test]
    fn test_view_to_index_contiguous() {
        let idx_gen = IndexGenerator::new();
        let view = View::contiguous(vec![Expr::Const(32), Expr::Const(64)]);

        let idx = idx_gen.view_to_index(&view);
        // Contiguous view produces an Add expression
        assert!(matches!(idx, AstNode::Add(_, _)));
    }
}
