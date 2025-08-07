pub mod ast;
pub mod graph;

use crate::{
    ast::{AstNode, AstOp},
    rule,
};

use self::ast::{AstRewriter, RewriteRule};
use std::rc::Rc;

/// Returns an `AstRewriter` configured with a set of algebraic simplification rules.
/// This version performs checks inside the rewriter function.
pub fn algebraic_simplification() -> AstRewriter {
    let rules: Vec<Rc<RewriteRule>> = vec![
        // x + 0 -> x  OR  0 + x -> x
        rule!("add_zero", |a, b| a.clone() + b.clone() => {
            if let AstOp::Const(c) = &b.op {
                if c.is_zero() {
                    return a;
                }
            }
            if let AstOp::Const(c) = &a.op {
                if c.is_zero() {
                    return b;
                }
            }
            a + b
        }),
        // x * 1 -> x  OR  1 * x -> x
        rule!("mul_one", |a, b| a.clone() * b.clone() => {
            if let AstOp::Const(c) = &b.op {
                if c.is_one() {
                    return a;
                }
            }
            if let AstOp::Const(c) = &a.op {
                if c.is_one() {
                    return b;
                }
            }
            a * b
        }),
        // x * 0 -> 0  OR  0 * x -> 0
        rule!("mul_zero", |a, b| a.clone() * b.clone() => {
            if let AstOp::Const(c) = &b.op {
                if c.is_zero() {
                    return AstNode::from(0.0f32).cast(a.dtype);
                }
            }
            if let AstOp::Const(c) = &a.op {
                if c.is_zero() {
                    return AstNode::from(0.0f32).cast(b.dtype);
                }
            }
            a * b
        }),
        // x - 0 -> x
        rule!("sub_zero", |a, b| a.clone() - b.clone() => {
            if let AstOp::Const(c) = &b.op {
                if c.is_zero() {
                    return a;
                }
            }
            a - b
        }),
        // x / 1 -> x
        rule!("div_one", |a, b| a.clone() / b.clone() => {
            if let AstOp::Const(c) = &b.op {
                if c.is_one() {
                    return a;
                }
            }
            a / b
        }),
    ];
    AstRewriter::new(rules)
}
