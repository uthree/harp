//! Loop structure generation for kernel bodies
//!
//! This module generates nested loop structures from shape information,
//! creating the iteration pattern needed for tensor operations.

use crate::DType;
use crate::ast::{AstNode, Literal, Mutability, ParallelInfo, Scope};
use crate::graph::shape::Expr;

use super::index_gen::IndexGenerator;

/// Loop generator for creating nested iteration structures
pub struct LoopGenerator {
    index_gen: IndexGenerator,
}

impl Default for LoopGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopGenerator {
    /// Create a new loop generator
    pub fn new() -> Self {
        Self {
            index_gen: IndexGenerator::new(),
        }
    }

    /// Create loop generator with custom index generator
    pub fn with_index_gen(index_gen: IndexGenerator) -> Self {
        Self { index_gen }
    }

    /// Get the index generator
    pub fn index_gen(&self) -> &IndexGenerator {
        &self.index_gen
    }

    /// Generate nested loops for the given shape
    ///
    /// Creates a nested Range structure like:
    /// ```text
    /// Range { var: "ridx0", stop: shape[0],
    ///   body: Range { var: "ridx1", stop: shape[1],
    ///     body: <inner_body>
    ///   }
    /// }
    /// ```
    pub fn generate_loops(&self, shape: &[Expr], inner_body: AstNode) -> AstNode {
        let bounds = self.index_gen.shape_to_bounds(shape);

        // Build from inside out
        let mut result = inner_body;
        for (var, stop) in bounds.into_iter().rev() {
            result = AstNode::Range {
                var,
                start: Box::new(AstNode::Const(Literal::I64(0))),
                stop: Box::new(stop),
                step: Box::new(AstNode::Const(Literal::I64(1))),
                body: Box::new(result),
                parallel: ParallelInfo::default(),
            };
        }

        result
    }

    /// Generate loops with a specific axis marked for reduction
    ///
    /// The reduction axis will have its loop variable available for accumulation.
    /// The `acc_var` and `acc_dtype` parameters are used to declare the accumulator
    /// variable in the reduce block's scope.
    pub fn generate_loops_with_reduce(
        &self,
        shape: &[Expr],
        reduce_axis: usize,
        pre_reduce: AstNode,
        reduce_body: AstNode,
        post_reduce: AstNode,
        acc_var: &str,
        acc_dtype: &DType,
    ) -> AstNode {
        let bounds = self.index_gen.shape_to_bounds(shape);

        // Split loops into outer (before reduce), reduce, and inner (after reduce)
        let (outer_bounds, rest) = bounds.split_at(reduce_axis);
        let (reduce_bounds, inner_bounds) = rest.split_first().unwrap();

        // Build inner loops (if any)
        let mut inner_result = reduce_body;
        for (var, stop) in inner_bounds.iter().rev() {
            inner_result = AstNode::Range {
                var: var.clone(),
                start: Box::new(AstNode::Const(Literal::I64(0))),
                stop: Box::new(stop.clone()),
                step: Box::new(AstNode::Const(Literal::I64(1))),
                body: Box::new(inner_result),
                parallel: ParallelInfo::default(),
            };
        }

        // Build reduction loop with pre/post
        let reduce_loop = AstNode::Range {
            var: reduce_bounds.0.clone(),
            start: Box::new(AstNode::Const(Literal::I64(0))),
            stop: Box::new(reduce_bounds.1.clone()),
            step: Box::new(AstNode::Const(Literal::I64(1))),
            body: Box::new(inner_result),
            parallel: ParallelInfo::default(),
        };

        // Create scope with acc variable declared
        let mut scope = Scope::new();
        scope
            .declare(acc_var.to_string(), acc_dtype.clone(), Mutability::Mutable)
            .expect("Failed to declare acc variable in reduce scope");

        // Combine pre, reduce loop, post using Block
        let reduce_block = AstNode::Block {
            statements: vec![pre_reduce, reduce_loop, post_reduce],
            scope: Box::new(scope),
        };

        // Build outer loops
        let mut result = reduce_block;
        for (var, stop) in outer_bounds.iter().rev() {
            result = AstNode::Range {
                var: var.clone(),
                start: Box::new(AstNode::Const(Literal::I64(0))),
                stop: Box::new(stop.clone()),
                step: Box::new(AstNode::Const(Literal::I64(1))),
                body: Box::new(result),
                parallel: ParallelInfo::default(),
            };
        }

        result
    }

    /// Generate a simple elementwise loop pattern
    ///
    /// For shape [N, M], generates:
    /// ```text
    /// for ridx0 in 0..N:
    ///     for ridx1 in 0..M:
    ///         output[idx] = f(input[idx])
    /// ```
    pub fn generate_elementwise(
        &self,
        shape: &[Expr],
        output_ptr: AstNode,
        output_idx: AstNode,
        element_expr: AstNode,
    ) -> AstNode {
        let store = AstNode::Store {
            ptr: Box::new(output_ptr),
            offset: Box::new(output_idx),
            value: Box::new(element_expr),
        };
        self.generate_loops(shape, store)
    }

    /// Generate a reduction loop pattern
    ///
    /// For shape [N, M] with reduce on axis 1:
    /// ```text
    /// for ridx0 in 0..N:
    ///     float acc;
    ///     acc = identity
    ///     for ridx1 in 0..M:
    ///         acc = combine(acc, input[idx])
    ///     output[ridx0] = acc
    /// ```
    pub fn generate_reduce(
        &self,
        shape: &[Expr],
        reduce_axis: usize,
        output_ptr: AstNode,
        output_idx: AstNode,
        acc_var: &str,
        acc_dtype: &DType,
        identity: AstNode,
        combine_expr: AstNode,
    ) -> AstNode {
        let pre = AstNode::Assign {
            var: acc_var.to_string(),
            value: Box::new(identity),
        };

        let reduce_body = AstNode::Assign {
            var: acc_var.to_string(),
            value: Box::new(combine_expr),
        };

        let post = AstNode::Store {
            ptr: Box::new(output_ptr),
            offset: Box::new(output_idx),
            value: Box::new(AstNode::Var(acc_var.to_string())),
        };

        self.generate_loops_with_reduce(
            shape,
            reduce_axis,
            pre,
            reduce_body,
            post,
            acc_var,
            acc_dtype,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_simple_loops() {
        let loop_gen = LoopGenerator::new();
        let shape = vec![Expr::Const(32), Expr::Const(64)];
        let body = AstNode::Const(Literal::I64(0)); // placeholder

        let loops = loop_gen.generate_loops(&shape, body);

        // Should be nested Range
        assert!(matches!(loops, AstNode::Range { .. }));
    }

    #[test]
    fn test_generate_elementwise() {
        let loop_gen = LoopGenerator::new();
        let shape = vec![Expr::Const(32)];
        let output_ptr = AstNode::Var("output".to_string());
        let output_idx = AstNode::Var("ridx0".to_string());
        let expr = AstNode::Var("x".to_string());

        let ast = loop_gen.generate_elementwise(&shape, output_ptr, output_idx, expr);

        assert!(matches!(ast, AstNode::Range { .. }));
    }

    #[test]
    fn test_generate_reduce() {
        let loop_gen = LoopGenerator::new();
        let shape = vec![Expr::Const(32), Expr::Const(64)];
        let output_ptr = AstNode::Var("output".to_string());
        let output_idx = AstNode::Var("ridx0".to_string());
        let identity = AstNode::Const(Literal::F32(0.0));
        let combine = AstNode::Add(
            Box::new(AstNode::Var("acc".to_string())),
            Box::new(AstNode::Var("val".to_string())),
        );

        let ast = loop_gen.generate_reduce(
            &shape,
            1,
            output_ptr,
            output_idx,
            "acc",
            &DType::F32,
            identity,
            combine,
        );

        // Should be a Range containing a Block
        assert!(matches!(ast, AstNode::Range { .. }));
    }
}
