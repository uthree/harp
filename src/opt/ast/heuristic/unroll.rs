use std::collections::HashSet;

use crate::ast::{AstNode, AstOp};
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester that proposes loop unrolling transformations.
pub struct UnrollSuggester {
    pub unroll_factor: usize,
}

impl UnrollSuggester {
    pub fn new(unroll_factor: usize) -> Self {
        Self { unroll_factor }
    }
}

impl Default for UnrollSuggester {
    fn default() -> Self {
        Self::new(4)
    }
}

impl RewriteSuggester for UnrollSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = vec![];
        self.suggest_rec(node, &mut suggestions);
        suggestions
    }
}

impl UnrollSuggester {
    fn suggest_rec(&self, node: &AstNode, suggestions: &mut Vec<AstNode>) {
        // First, recurse on children to find potential transformations deeper in the tree.
        for child in &node.src {
            self.suggest_rec(child, suggestions);
        }

        // Then, check if the current node itself is a candidate for unrolling.
        if let AstOp::Range { counter, step } = &node.op
            && *step == 1 {
                // Only unroll loops with a step of 1 for now.
                let start = node.src[0].clone();
                let loop_limit = node.src[1].clone();
                let loop_body = node.src[2].clone();

                // Main unrolled loop
                let unrolled_limit = loop_limit.clone() / AstNode::from(self.unroll_factor);
                let unrolled_body_stmts: Vec<AstNode> = (0..self.unroll_factor)
                    .map(|i| {
                        let new_counter = AstNode::var(counter, node.dtype.clone())
                            * AstNode::from(self.unroll_factor)
                            + AstNode::from(i);
                        replace_var_in_expr(&loop_body, counter, &new_counter, &mut HashSet::new())
                    })
                    .collect();
                let unrolled_body = AstNode::block(unrolled_body_stmts);
                let main_loop = AstNode::range(
                    counter,
                    self.unroll_factor as isize,
                    start,
                    unrolled_limit,
                    unrolled_body,
                );

                // Remainder loop
                let remainder_start = (loop_limit.clone() / AstNode::from(self.unroll_factor))
                    * AstNode::from(self.unroll_factor);
                let remainder_loop =
                    AstNode::range(counter, 1, remainder_start, loop_limit, loop_body.clone());

                let unrolled_sequence = AstNode::block(vec![main_loop, remainder_loop]);
                suggestions.push(unrolled_sequence);
            }
    }
}

/// Replaces all occurrences of a variable within an expression, avoiding capture.
fn replace_var_in_expr(
    expr: &AstNode,
    var_name: &str,
    new_expr: &AstNode,
    bound_vars: &mut HashSet<String>,
) -> AstNode {
    if let AstOp::Var(name) = &expr.op
        && name == var_name && !bound_vars.contains(var_name) {
            return new_expr.clone();
        }

    // If we encounter a new scope (e.g., a new loop), add its counter to the bound_vars set.
    if let AstOp::Range { counter, .. } = &expr.op {
        bound_vars.insert(counter.clone());
    }

    let new_srcs = expr
        .src
        .iter()
        .map(|child| replace_var_in_expr(child, var_name, new_expr, bound_vars))
        .collect();

    // After processing the children, if we added a variable to the set, remove it.
    if let AstOp::Range { counter, .. } = &expr.op {
        bound_vars.remove(counter);
    }

    AstNode::_new(expr.op.clone(), new_srcs, expr.dtype.clone())
}
